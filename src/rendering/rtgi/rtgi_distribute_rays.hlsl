#pragma once

#include "rtgi_distribute_rays.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtgiDistributeRaysPush rtgi_distribute_rays_push;

// Discretionary ray distribution weighting (draining path):
//   1 = squared deficit (disc_i^2) — strongly prefers the freshest disocclusions.
//   0 = linear deficit (disc_i) — near-flat split across all under-converged pixels.
#define RTGI_RAY_PRIORITY_SQUARED 1

uint ray_priority_weight(uint disc_i)
{
#if RTGI_RAY_PRIORITY_SQUARED
    return disc_i * disc_i;
#else
    return disc_i;
#endif
}

// == Groupshared state =======================================================
// One workgroup covers exactly one 8×8 tile.
static const uint TILE_THREADS = RTGI_DISTRIBUTE_RAYS_X * RTGI_DISTRIBUTE_RAYS_Y; // 64

// == Tile groupshared SLOT ===================================================
// Each pixel addresses groupshared by its plain row-major linear index. (The base distribution is a
// per-pixel Bayer dither and the extra distribution's only order-dependent part is a ±1 rounding
// remainder, so no slot permutation is needed to keep either spatially even.)
uint2 tile_slot_to_pixel(uint slot)
{
    return uint2(slot % RTGI_DISTRIBUTE_RAYS_X, slot / RTGI_DISTRIBUTE_RAYS_X);
}

// 8x8 ordered-dither (Bayer) rank in [0,63]. Thresholding rank/64 < f selects a spatially EVEN ~f fraction
// for any f — a perfect checkerboard at f=0.5. Recursive 2x2 base [[0,3],[2,1]] with the finest level in
// the HIGH bits, so low thresholds still spread at the finest (checkerboard) frequency. Used for the base
// ray distribution so a half-rate tile lays its base rays out as a checkerboard instead of a coarse band.
uint bayer_m2(uint a, uint b) { return 2u * ((a ^ b) & 1u) + (a & 1u); } // 0..3, <2 == diagonal checkerboard
uint bayer_8x8(uint x, uint y)
{
    return 16u * bayer_m2(x, y) + 4u * bayer_m2(x >> 1u, y >> 1u) + bayer_m2(x >> 2u, y >> 2u);
}

groupshared uint gs_desired[TILE_THREADS]; // desired total rays per thread (0=sky/oob, 1+=geo)
groupshared uint gs_actual[TILE_THREADS];  // rays allocated to each thread after budgeting
groupshared uint gs_offset[TILE_THREADS];  // exclusive prefix sum of gs_actual within the tile
groupshared uint gs_base_offset;           // global ray list offset reserved for this tile
groupshared uint gs_tile_total;            // total rays allocated to this tile (for per-tile debug viz)

// NOTE: the whole budgeting + prefix sum + reservation runs single-threaded on thread 0. It is all
// trivial integer work over 64 elements, so a serial pass is cheaper than a parallel Hillis-Steele
// scan once you count the ~15 GroupMemoryBarrierWithGroupSync a parallel scan needs. This shader now
// synchronizes with exactly TWO barriers per tile: one after every thread publishes its gs_desired,
// and one after thread 0 publishes gs_actual / gs_offset / gs_base_offset.

[shader("compute")]
[numthreads(RTGI_DISTRIBUTE_RAYS_X, RTGI_DISTRIBUTE_RAYS_Y, 1)]
func entry_distribute_rays(uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_distribute_rays_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    const uint flat_id = gtid.y * RTGI_DISTRIBUTE_RAYS_X + gtid.x; // linear == hardware lane (wave logic)
    const uint slot    = flat_id;                                // groupshared slot == row-major linear index
    const uint2 pixel_xy = gid * uint2(RTGI_DISTRIBUTE_RAYS_X, RTGI_DISTRIBUTE_RAYS_Y) + gtid;
    const bool in_bounds = all(pixel_xy < push.size);

    // == Per-pixel desired ray count =========================================
    float reprojected_sample_count = -1.0f;
    if (in_bounds)
        reprojected_sample_count = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[pixel_xy]);

    const bool is_geo = in_bounds && reprojected_sample_count >= 0.0f; // sky has reprojected_sample_count < 0
    const uint desired_total = is_geo ? calc_desired_ray_count(reprojected_sample_count, rtgi_settings.fast_convergence_samples) : 0u;
    gs_desired[slot] = desired_total;

    GroupMemoryBarrierWithGroupSync();

    // == Compute per-pixel ray counts (one wave) =============================
    // The per-tile budgeting is done cooperatively by a single wave instead of serially on lane 0. The
    // 64 tile elements are split epl = 64/wave_size per lane; sums use WaveActiveSum and the offset /
    // allocation scans use WavePrefixSum. Result is bit-identical to the old serial loop (see below).
    // Each distribution scales by a precomputed float reciprocal instead of a runtime integer divide;
    // every `target` is monotonic + capped, so each differenced alloc is >= 0 and the per-distribution
    // counts sum to exactly that distribution's tile budget (base + extra).
    if (flat_id < WaveGetLaneCount())
    {
        // ONE wave cooperatively does all the budgeting (the rest of the tile's threads idle here). The
        // 64 tile elements are split across the wave: epl = 64 / wave_size elements per lane (2 on a
        // 32-wide wave, 1 on a 64-wide wave), so this adapts to the hardware wave size automatically.
        // Lane l owns the contiguous block i in [l*epl, (l+1)*epl), which preserves the original i=0..63
        // ordering as lane-major-then-within-block — required so the weight prefix (which decides WHICH
        // pixels win rounded rays) is bit-identical to the old serial loop. Reductions use WaveActiveSum;
        // the offset/allocation scans use WavePrefixSum of per-lane block totals + a serial in-block scan.
        // (Relies on the standard linear SV_GroupIndex->lane mapping so lane == flat_id in wave 0.)
        const uint wave_size = WaveGetLaneCount();
        const uint lane      = flat_id;
        const uint epl       = TILE_THREADS / wave_size;

        const float ray_pct = clamp(rtgi_settings.ray_percentage, 0.0f, float(RTGI_RAY_LIST_CAPACITY_MUL));
        uint tile_total = 0u; // final reserved ray count (wave-uniform after WaveActiveSum)

        if (!rtgi_settings.use_ray_redistribution)
        {
            // Redistribution off: every geometry pixel traces the exact same fixed number of rays,
            // max(floor(ray_budget), 1), with no demand weighting, tile proportionality, or rotation.
            const uint fixed_rays = calc_fixed_rays_per_pixel(ray_pct);
            uint lane_sum = 0u;
            for (uint e = 0u; e < epl; ++e)
            {
                const uint i = lane * epl + e;
                const uint actual_i = gs_desired[i] > 0u ? fixed_rays : 0u;
                gs_actual[i] = actual_i;
                lane_sum += actual_i;
            }
            const uint lane_base = WavePrefixSum(lane_sum); // exclusive prefix of block totals
            tile_total = WaveActiveSum(lane_sum);
            uint run = lane_base;
            for (uint e = 0u; e < epl; ++e)
            {
                const uint i = lane * epl + e;
                gs_offset[i] = run;
                run += gs_actual[i];
            }
        }
        else
        {
            // Redistribution on: a BASE distribution + an EXTRA distribution, both drained by the SAME
            // global fraction so the base coverage is uniform across the whole screen.
            //
            //  * global_ratio = fraction of ALL demand the frame budget can afford. A converged region
            //    (only base demand) is drained by exactly this factor: if the screen can afford 2/3 of the
            //    rays, ~2/3 of converged pixels get their base ray.
            //
            //  * BASE: every tile drains its base rays by that SAME global_ratio — a busy tile gives the
            //    same fraction of its pixels a base ray as a converged tile does (uniform base coverage,
            //    no tile getting full base coverage just because it also requested extras).
            //
            //  * EXTRA: the surplus (desired-1 per pixel), ALSO scaled by global_ratio, distributed by
            //    SQUARED deficit weight (freshest disocclusions win) and added ON TOP of the base rays.
            //
            // So a heavily-disoccluded tile still drains converged tiles across the screen (its extras pull
            // from the shared budget), but WITHIN any tile the base rays are never over-drained to feed the
            // extras — base and extra are separately budgeted, both at the uniform global fraction.
            const uint frame = rtgi_settings.animate_noise ? push.attach.globals.frame_index : 0u;
            const float min_budget = clamp(rtgi_settings.min_ray_budget, 0.0f, 1.0f);

            const uint total_halfres = push.size.x * push.size.y;
            const uint total_budget  = uint(float(total_halfres) * max(ray_pct, min_budget));
            const uint total_geo     = push.attach.ray_counters->total_geo_rays;
            const uint total_extra   = push.attach.ray_counters->total_extra_rays;
            const uint total_desired = total_geo + total_extra;
            // Global affordable fraction of ALL demand — the uniform drain factor applied to base AND extra.
            const float global_ratio = float(total_budget) / float(total_desired + 1u);
            const uint tile_id = gid.y * (push.size.x / RTGI_DISTRIBUTE_RAYS_X) + gid.x;

            // Branch is wave-uniform (all inputs uniform), so the wave intrinsics below run in uniform flow.
            if (total_budget >= total_desired)
            {
                // Fast path (not draining): every geometry pixel gets its full desired count.
                uint lane_sum = 0u;
                for (uint e = 0u; e < epl; ++e)
                {
                    const uint i = lane * epl + e;
                    const uint actual_i = gs_desired[i];
                    gs_actual[i] = actual_i;
                    lane_sum += actual_i;
                }
                const uint lane_base = WavePrefixSum(lane_sum);
                tile_total = WaveActiveSum(lane_sum);
                uint run = lane_base;
                for (uint e = 0u; e < epl; ++e)
                {
                    const uint i = lane * epl + e;
                    gs_offset[i] = run;
                    run += gs_actual[i];
                }
            }
            else
            {
                // Draining path: base and extra each drained by the SAME global_ratio (uniform base
                // coverage across the screen), extra added on top.
                //   base_i  = 1 for a geo pixel (weight is uniform -> an even split of the base budget).
                //   extra_i = desired - 1  (weight is extra_i^2 -> squared-deficit priority).
                // Reduction: geo-pixel count (base demand), extra demand, squared-extra weight.
                uint lane_base_cnt = 0u; // geo pixels in this lane's block (== base demand + base weight)
                uint lane_extra_d  = 0u; // sum of extra_i        (sizes the extra budget)
                uint lane_extra_w  = 0u; // sum of extra_i^2      (priority weight)
                for (uint e = 0u; e < epl; ++e)
                {
                    const uint i = lane * epl + e;
                    const uint d = gs_desired[i];
                    const uint base_i  = d > 0u ? 1u : 0u;
                    const uint extra_i = d - base_i; // 0 for sky, desired-1 for geo
                    lane_base_cnt += base_i;
                    lane_extra_d  += extra_i;
                    lane_extra_w  += ray_priority_weight(extra_i);
                }
                const uint tile_base_cnt  = WaveActiveSum(lane_base_cnt);
                const uint tile_extra_dem = WaveActiveSum(lane_extra_d);
                const uint tile_extra_wgt = WaveActiveSum(lane_extra_w);
                // Both budgets scaled by the SAME global fraction: base coverage is uniform across tiles,
                // extra is the surplus added on top. Each capped at its own demand.
                // Base coverage floored at min_ray_budget so every tile always gets at least that rate.
                const uint tile_base_budget  = min(uint(float(tile_base_cnt)  * max(global_ratio, min_budget)), tile_base_cnt);
                const uint tile_extra_budget = min(uint(float(tile_extra_dem) * global_ratio), tile_extra_dem);

                // BASE distribution: ordered-dither (Bayer) threshold. A pixel gets its base ray when its
                // Bayer rank falls under the tile's base fraction — spatially EVEN at any fraction, and a
                // clean checkerboard at 0.5 (instead of the coarse band the strided even-split produced).
                // A per-frame rotation of the threshold cycles which pixels win over time.
                const float base_frac = tile_base_cnt > 0u ? float(tile_base_budget) / float(tile_base_cnt) : 0.0f;
                const float base_dither_rot = frac(float(frame) * 0.61803398875f);

                // EXTRA distribution: unchanged cumulative squared-deficit scan (phase rotates per frame/tile).
                const float extra_phase = frac(float(frame) * 0.61803398875f + float(tile_id) * 0.7548776662f + 0.5f);
                const float inv_extra_w = tile_extra_wgt > 0u ? rcp(float(tile_extra_wgt)) : 0.0f;
                const float extra_budget_f = float(tile_extra_budget);

                // Exclusive weight prefix at this lane's block start == the extra scan's `given` cursor.
                const uint lane_extra_w_base = WavePrefixSum(lane_extra_w);
                uint acc_extra_w = lane_extra_w_base;
                uint given_extra = tile_extra_wgt > 0u
                    ? min(uint(saturate(float(lane_extra_w_base) * inv_extra_w) * extra_budget_f + extra_phase), tile_extra_budget)
                    : 0u;
                uint lane_actual = 0u;
                for (uint e = 0u; e < epl; ++e)
                {
                    const uint i = lane * epl + e;
                    const uint d = gs_desired[i];
                    const uint base_i  = d > 0u ? 1u : 0u;
                    const uint extra_i = d - base_i;

                    // BASE: ordered-dither threshold on the pixel's Bayer rank -> spatially even (checkerboard
                    // at 0.5). One base ray if the (frame-rotated) rank is under the tile's base fraction.
                    const uint2 px = tile_slot_to_pixel(i);
                    const float bayer = float(bayer_8x8(px.x, px.y)) * (1.0f / 64.0f);
                    const uint base_alloc = (base_i > 0u && frac(bayer + base_dither_rot) < base_frac) ? 1u : 0u;

                    // EXTRA: squared-deficit priority, cap at extra_i so a lone fresh disocclusion can't be
                    // over-allocated. Any extra budget above a pixel's cap is left unspent.
                    acc_extra_w += ray_priority_weight(extra_i);
                    const uint target_extra = tile_extra_wgt > 0u
                        ? min(uint(saturate(float(acc_extra_w) * inv_extra_w) * extra_budget_f + extra_phase), tile_extra_budget)
                        : 0u;
                    const uint extra_alloc = min(target_extra - given_extra, extra_i);
                    given_extra = target_extra;

                    const uint actual_i = base_alloc + extra_alloc;
                    gs_actual[i] = actual_i;
                    lane_actual += actual_i;
                }
                const uint lane_actual_base = WavePrefixSum(lane_actual);
                tile_total = WaveActiveSum(lane_actual);
                uint run = lane_actual_base;
                for (uint e = 0u; e < epl; ++e)
                {
                    const uint i = lane * epl + e;
                    gs_offset[i] = run;
                    run += gs_actual[i];
                }
            }
        }

        // Reserve the tile's contiguous slice of the global ray list in one atomic (lane 0 only).
        if (lane == 0u)
        {
            uint old_count;
            InterlockedAdd(push.attach.ray_counters->ray_list_count, tile_total, old_count);
            gs_base_offset = old_count;
            gs_tile_total = tile_total;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // == Write ray list entries and per-pixel alloc ===========================
    const uint my_count  = gs_actual[slot];
    const uint my_offset = gs_base_offset + gs_offset[slot];

    // Hard capacity clamp: the ray list buffer holds exactly total_halfres entries. The budget math
    // keeps us within this, but guard the writes anyway so a rounding edge can never overflow the
    // buffer (which would be a GPU crash). Rays past capacity are simply dropped.
    const uint ray_list_capacity = push.size.x * push.size.y * RTGI_RAY_LIST_CAPACITY_MUL;
    uint clamped_count = my_count;
    if (my_offset >= ray_list_capacity) clamped_count = 0u;
    else if (my_offset + my_count > ray_list_capacity) clamped_count = ray_list_capacity - my_offset;

    if (in_bounds)
    {
        // Write the ray-list offset and the per-pixel ray count (the count is the single source of truth,
        // consumed by the blend pass and the pre-filter/temporal passes).
        push.attach.pixel_ray_alloc.get()[pixel_xy] = my_offset;
        push.attach.ray_count_image.get()[pixel_xy] = clamped_count;

        // Write one ray list entry per allocated ray for this pixel.
        const uint packed_xy = pixel_xy.x | (pixel_xy.y << 16u);
        for (uint s = 0u; s < clamped_count; s++)
        {
            push.attach.ray_list[my_offset + s] = RtgiRayEntry(packed_xy, s);
        }

        // == Debug visualizations ============================================
        RWTexture2D<float4> dbg = push.attach.debug_image.get();
        let debug_mode = push.attach.globals.settings.debug_draw_mode;
        const float debug_alpha = 1.0f + push.attach.globals.settings.debug_visualization_blend;
        // Per-tile: total rays for this 8x8 tile, normalized by the tile's max capacity.
        const float max_tile_rays = float(TILE_THREADS * RTGI_RAY_LIST_CAPACITY_MUL);
        if (debug_mode == DEBUG_DRAW_MODE_RTGI_RAYS_SHOT)
        {
            write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, pixel_xy, float4(Heatmap(float(clamped_count) / 8), debug_alpha), 2);
        }
        else if (debug_mode == DEBUG_DRAW_MODE_RTGI_RAYS_SHOT_PER_TILE)
        {
            // The overlay below is drawn directly into a fixed screen rect (two 64x64 patches at y in
            // [64,128)). write_debug_image floods the whole per-tile heatmap into one screen quadrant, which
            // would clobber that rect whenever debug_visualization_tile targets the top-left quadrant. So
            // replicate the helper's destination math and SKIP the flood for any source pixel that would
            // land inside the overlay rect — keeping both the heatmap and a guaranteed-visible overlay.
            const int  tile_slot = push.attach.globals.settings.debug_visualization_tile;
            uint dbg_w, dbg_h; dbg.GetDimensions(dbg_w, dbg_h);
            const uint2 slot_size = uint2(dbg_w, dbg_h) / 4u;
            const uint2 flood_dst = uint2(tile_slot % 4, tile_slot / 4) * slot_size + pixel_xy / 2u; // scale 2 => /4*2 == /2
            const bool  flood_hits_overlay = flood_dst.x < (2u * 64u + 8u) && flood_dst.y >= 64u && flood_dst.y < 128u;
            if (!flood_hits_overlay)
                write_debug_image(dbg, tile_slot, pixel_xy, float4(Heatmap(float(gs_tile_total) / max_tile_rays), debug_alpha), 2);

            // HACK: ray-allocation priority overlay for a SINGLE tile (0,0), magnified 8x into the
            // top-left of the screen (each of the 64 tile-pixels -> one 8x8 debug block => a 64x64 patch).
            // Shows the ORDER pixels win a discretionary ray as the budget grows, under uniform demand and
            // NO temporal phase (frame 0). This is the static, deterministic priority pattern the golden-
            // ratio phase rotates each frame to distribute coverage over time. A pixel's rank R means it
            // first receives a ray at budget R; so for a budget of e.g. 40, the pixels with rank <= 40 win.
            if (all(gid == uint2(0, 0)))
            {
                const uint scan_i = slot; // this pixel's permuted groupshared slot (== its position in the scan)
                // Uniform weight => cumulative membership at budget B is the differenced even split:
                //   member(i,B) = floor((i+1)*B/64) - floor(i*B/64) >= 1  (== the alloc scan with phase 0).
                // Rank = smallest budget B in [1,64] at which this pixel first gets a ray.
                uint rank = TILE_THREADS;
                for (uint B = 1u; B <= TILE_THREADS; ++B)
                {
                    const uint member = ((scan_i + 1u) * B) / TILE_THREADS - (scan_i * B) / TILE_THREADS;
                    if (member >= 1u) { rank = B; break; }
                }
                const float t = float(rank - 1u) / float(TILE_THREADS - 1u);
                const float4 prio_col = float4(Heatmap(t), debug_alpha);
                // gtid == tile-local pixel coords for tile (0,0); nudged down 64 display px so the overlay
                // clears the top edge / other HUD.
                const uint2 block_origin = gtid * 8u + uint2(0u, 64u);
                for (uint yy = 0u; yy < 8u; ++yy)
                    for (uint xx = 0u; xx < 8u; ++xx)
                        dbg[block_origin + uint2(xx, yy)] = prio_col;

                // SECOND patch, right of the first (8px gap): actual ray COUNT each pixel gets for a full
                // tile under the current ray_percentage setting, uniform demand, no phase. The tile's budget
                // is 64 * ray_percentage rays; the same even-split scan hands each pixel floor/ceil of the
                // average. Colored 0..RTGI_RAY_LIST_CAPACITY_MUL rays.
                const float ray_pct_dbg = clamp(rtgi_settings.ray_percentage, 0.0f, float(RTGI_RAY_LIST_CAPACITY_MUL));
                const uint  tile_budget = uint(float(TILE_THREADS) * ray_pct_dbg);
                const uint  count_i     = ((scan_i + 1u) * tile_budget) / TILE_THREADS - (scan_i * tile_budget) / TILE_THREADS;
                const float ct = float(count_i) / float(RTGI_RAY_LIST_CAPACITY_MUL);
                const float4 count_col = float4(Heatmap(ct), debug_alpha);
                const uint2 count_origin = block_origin + uint2(TILE_THREADS + 8u, 0u); // 64 + 8px gap to the right
                for (uint yy2 = 0u; yy2 < 8u; ++yy2)
                    for (uint xx2 = 0u; xx2 < 8u; ++xx2)
                        dbg[count_origin + uint2(xx2, yy2)] = count_col;
            }
        }

        // write_debug_image(dbg, 0, pixel_xy, float4(Heatmap(float(reprojected_sample_count) / 32), 2), 2);
        // write_debug_image(dbg, 1, pixel_xy, float4(Heatmap(float(gs_tile_total) / max_tile_rays), 2), 2);
    }
}
