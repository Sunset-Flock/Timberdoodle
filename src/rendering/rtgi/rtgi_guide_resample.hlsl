#pragma once

#include "rtgi_guide_resample.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/debug.glsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiGuideResampleHorizontalPush rtgi_guide_resample_horizontal_push;
[[vk::push_constant]] RtgiGuideResampleVerticalPush rtgi_guide_resample_vertical_push;
[[vk::push_constant]] RtgiGuideResolvePush rtgi_guide_resolve_push;

// === RTGI Guide Resample =========================================================================
// Turns the sparse pioneer trace (pioneer_hit_y, one ray per pioneer cell -- see pioneer_ray_gen in
// rtgi_trace_diffuse.hlsl) into a per-half-res-pixel guide direction (guide_sh_y / guide_confidence),
// via three passes: horizontal resample, vertical resample, resolve.
//
// RECONNECTION, NOT A BAKED DIRECTION
// Every candidate carried through the horizontal and vertical passes is a (hit position X, brightness
// Y) pair, never a direction. The pioneer ray that found X was cast from ITS OWN pixel, generally a
// different surface point than whichever half-res pixel eventually asks for a guide -- baking that
// ray's direction in early would only be correct for the one pixel it was cast from. Y needs no such
// caveat: the traced surface is Lambertian (this whole pipeline is diffuse-only GI), so its exitant
// radiance is the SAME toward every direction, and stays valid for any receiver. So the horizontal and
// vertical passes pick candidates purely by position-gated brightness -- they never look at or need a
// direction at all. Only entry_guide_resolve (the third pass) ever turns a winning (X, Y) into a real
// direction: normalize(X - receiver_position), computed fresh for whichever half-res pixel is actually
// asking. That's the reconnection -- the direction is re-derived per receiver instead of copied from
// the original pioneer ray, so one pioneer sample can correctly serve every nearby pixel that reuses
// it, not just the one it came from.
//
// WHY THE SPLIT (H THEN V) STILL GIVES THE SAME ODDS AS ONE FLAT 2D PICK
// The horizontal and vertical passes are both weighted random picks (RIS: reservoir sampling by
// importance) over a WINDOW x WINDOW neighborhood of pioneer cells, done as WINDOW + WINDOW taps
// instead of WINDOW^2 by splitting into a row pass and a column pass. This costs nothing in
// correctness: the horizontal pass reduces each row of candidates c_0..c_n (weights w_0..w_n) to one
// picked value v_row, kept with probability w_i / W_row (W_row = sum of the row's weights) -- and
// outputs W_row alongside v_row. The vertical pass then treats each row's v_row as a single candidate
// weighted by W_row, and does a SECOND weighted pick among the rows. The chance the final result is
// candidate c_i sitting in row r works out to (W_r / sum_of_all_W) * (w_i / W_r) = w_i / sum_of_all_W
// -- exactly the odds c_i would have had in one flat 2D pick over the whole window. No approximation,
// just a cheaper way to draw from the same distribution -- standard streaming/reservoir-combination
// math.

// === Horizontal pass: dispatched at pioneer resolution ===================
[shader("compute")]
[numthreads(RTGI_GUIDE_RESAMPLE_X, RTGI_GUIDE_RESAMPLE_Y, 1)]
func entry_guide_resample_horizontal(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_guide_resample_horizontal_push;
    const int2 pioneer_grid_size = int2(push.size); // this pass IS dispatched at pioneer resolution
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    const CameraInfo camera = push.attach.globals.view_camera;
    const uint2 half_res_size = push.attach.globals.settings.render_target_size >> 1u;
    const float2 inv_half_res_size = rcp(float2(half_res_size));

    // Same rotation formula pioneer_ray_gen (rtgi_trace_diffuse.hlsl) uses -- MUST stay in sync. Gated by
    // animate_noise like pioneer_ray_gen's own rotation, so a frozen frame maps to the SAME half-res pixel
    // pioneer_ray_gen actually traced that frame. trunk_flt_frame_index, not raw frame_index -- RTGI
    // convention (see globals.inl); mod-DIV unaffected since DIV divides the 4096 truncation period.
    const uint rotation_frame = push.attach.globals.rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    const uint2 rotation = uint2(
        rotation_frame % RTGI_GUIDE_PIONEER_GRID_DIV,
        (rotation_frame / RTGI_GUIDE_PIONEER_GRID_DIV) % RTGI_GUIDE_PIONEER_GRID_DIV);

    // This pioneer cell's own mapped half-res pixel -- the reference geometry every candidate in this
    // row is gated against (not the eventual half-res OUTPUT pixel, which the vertical pass gates
    // against instead; see that pass).
    const uint2 center_pixel = dtid * RTGI_GUIDE_PIONEER_GRID_DIV + rotation;
    if (any(center_pixel >= half_res_size) || push.attach.view_cam_half_res_depth.get()[center_pixel] == 0.0f)
    {
        push.attach.h_resample_hit_y.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        push.attach.h_resample_weight.get()[dtid] = float2(0.0f, 0.0f);
        return;
    }
    const PixelData center = calc_pixel_data(center_pixel, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
    const float center_pixel_width_ws = calc_pixel_width_ws(inv_half_res_size, camera.near_plane, center.ndc.z);

    const uint h_frame_seed = push.attach.globals.rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) * 257u : 0u;
    rand_seed(dtid.x * 9629u + dtid.y * 10069u + h_frame_seed + 11u);

    const float IMPORTANCE_FLOOR = 1e-6f;
    // RTGI_GUIDE_RESAMPLE_WINDOW is in DENOISER (half-res) pixels -- convert to a pioneer-cell tap count
    // here so the on-screen reach stays the same regardless of RTGI_GUIDE_PIONEER_GRID_DIV (see the define).
    const int guide_resample_taps = max(1, int(RTGI_GUIDE_RESAMPLE_WINDOW) / int(RTGI_GUIDE_PIONEER_GRID_DIV));
    const int base_x = int(dtid.x) - (guide_resample_taps / 2) * RTGI_GUIDE_RESAMPLE_STRIDE;
    float r_wsum = 0.0f;
    float valid_count = 0.0f;
    float4 r_held_hit_y = float4(0.0f, 0.0f, 0.0f, 0.0f); // .xyz = hit world pos, .w = brightness Y

    [loop] for (int i = 0; i < guide_resample_taps; ++i)
    {
        const int cell_x = base_x + i * RTGI_GUIDE_RESAMPLE_STRIDE;
        if (cell_x < 0 || cell_x >= pioneer_grid_size.x)
        {
            continue;
        }
        const uint2 cell = uint2(uint(cell_x), dtid.y);
        const float4 candidate_hit_y = push.attach.pioneer_hit_y.get()[cell];
        if (candidate_hit_y.w < 0.0f)
        {
            continue; // sky / out-of-bounds pioneer cell
        }

        const uint2 candidate_pixel = cell * RTGI_GUIDE_PIONEER_GRID_DIV + rotation;
        const PixelData candidate = calc_pixel_data(candidate_pixel, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());

        // Validity gate against THIS cell's own geometry (`center`) -- not folded into the selection
        // weight. Selection odds among eligible candidates are radiance only.
#if RTGI_GUIDE_RESAMPLE_DISTANCE_ONLY
        // Hard world-space distance cutoff only, no coplanarity/normal check -- see the define's comment
        // in rtgi_guide_resample.inl. Reference pixel width is whichever of the two points is closer to
        // camera (smaller footprint), so the cutoff can't loosen just because the far point sits further out.
        const float candidate_pixel_width_ws = calc_pixel_width_ws(inv_half_res_size, camera.near_plane, candidate.ndc.z);
        const bool valid = distance(center.position_ws, candidate.position_ws) <= RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD * min(center_pixel_width_ws, candidate_pixel_width_ws);
#else
        // px_threshold loosened to 8x (default 1.2x) -- the resample now reaches WINDOW*STRIDE pioneer
        // cells out, so the plane-distance acceptance needs to stay loose enough to still admit
        // distant-but-coplanar taps.
        const bool valid = (calc_similar_surface_weight(rcp(center_pixel_width_ws), center.position_ws, center.normal_ws, candidate.position_ws, candidate.normal_ws, RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD)
                           * calc_similar_normal_weight(center.normal_ws, candidate.normal_ws)) > 0.0f;
#endif
        if (!valid)
        {
            continue;
        }

        // log1p, not raw radiance -- a bright pioneer hit (direct light, specular flash) would otherwise
        // dominate the reservoir's odds near-deterministically and get RIS-picked across the whole reach
        // of this pass (no firefly clamp precedes the pioneer trace, unlike the main ray_result path).
        // log compresses that dynamic range so brightness still steers the pick without one outlier owning it.
        const float importance = max(log(candidate_hit_y.w + 1.0f), IMPORTANCE_FLOOR);
        r_wsum += importance;
        valid_count += 1.0f;
        if (rand() * r_wsum < importance)
        {
            r_held_hit_y = candidate_hit_y;
        }
    }

    push.attach.h_resample_hit_y.get()[dtid] = r_held_hit_y;
    push.attach.h_resample_weight.get()[dtid] = float2(r_wsum, valid_count);
}

// === Vertical pass =======================
[shader("compute")]
[numthreads(RTGI_GUIDE_RESAMPLE_X, RTGI_GUIDE_RESAMPLE_Y, 1)]
func entry_guide_resample_vertical(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_guide_resample_vertical_push;
    const int2 pioneer_grid_size = int2(push.size); // this pass IS dispatched at pioneer resolution
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    const CameraInfo camera = push.attach.globals.view_camera;
    const uint2 half_res_size = push.attach.globals.settings.render_target_size >> 1u;
    const float2 inv_half_res_size = rcp(float2(half_res_size));

    // Same rotation formula pioneer_ray_gen / the horizontal pass use -- MUST stay in sync. Gated by
    // animate_noise, trunk_flt_frame_index not raw frame_index -- see the horizontal pass for why.
    const uint rotation_frame = push.attach.globals.rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    const uint2 rotation = uint2(
        rotation_frame % RTGI_GUIDE_PIONEER_GRID_DIV,
        (rotation_frame / RTGI_GUIDE_PIONEER_GRID_DIV) % RTGI_GUIDE_PIONEER_GRID_DIV);

    // This pioneer cell's own mapped half-res pixel -- same reference geometry the horizontal pass used
    // for this exact cell (dtid here is identical to the horizontal pass's dtid).
    const uint2 center_pixel = dtid * RTGI_GUIDE_PIONEER_GRID_DIV + rotation;
    if (any(center_pixel >= half_res_size) || push.attach.view_cam_half_res_depth.get()[center_pixel] == 0.0f)
    {
        push.attach.pioneer_guide_hit_y.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        push.attach.pioneer_guide_confidence.get()[dtid] = 0.0f;
        return;
    }
    const PixelData center = calc_pixel_data(center_pixel, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
    const float center_pixel_width_ws = calc_pixel_width_ws(inv_half_res_size, camera.near_plane, center.ndc.z);

    const uint v_frame_seed = push.attach.globals.rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) * 257u : 0u;
    rand_seed(dtid.x * 9629u + dtid.y * 10069u + v_frame_seed + 29u);

    // RTGI_GUIDE_RESAMPLE_WINDOW is in DENOISER (half-res) pixels -- convert to a pioneer-cell tap count
    // here (same formula as the horizontal pass) so the on-screen reach stays the same regardless of
    // RTGI_GUIDE_PIONEER_GRID_DIV.
    const int guide_resample_taps = max(1, int(RTGI_GUIDE_RESAMPLE_WINDOW) / int(RTGI_GUIDE_PIONEER_GRID_DIV));
    const int base_y = int(dtid.y) - (guide_resample_taps / 2) * RTGI_GUIDE_RESAMPLE_STRIDE;
    float v_wsum = 0.0f;
    float v_valid_count = 0.0f;
    float4 v_held_hit_y = float4(0.0f, 0.0f, 0.0f, 0.0f); // .xyz = hit world pos, .w = brightness Y

    [loop] for (int i = 0; i < guide_resample_taps; ++i)
    {
        const int row = base_y + i * RTGI_GUIDE_RESAMPLE_STRIDE;
        if (row < 0 || row >= pioneer_grid_size.y)
        {
            continue;
        }
        const uint2 cell = uint2(dtid.x, uint(row));
        const float2 row_weight = push.attach.h_resample_weight.get()[cell]; // (W_row, valid_count_row)
        if (row_weight.x <= 0.0f)
        {
            continue; // this row had no valid candidate at all
        }

        // Row's representative geometry (same cell -> half-res-pixel mapping the horizontal pass used as
        // ITS OWN reference), gated here against THIS cell's own reference (`center`) instead.
        const uint2 row_center_pixel = cell * RTGI_GUIDE_PIONEER_GRID_DIV + rotation;
        if (any(row_center_pixel >= half_res_size))
        {
            continue;
        }
        if (push.attach.view_cam_half_res_depth.get()[row_center_pixel] == 0.0f)
        {
            continue;
        }
        const PixelData row_center = calc_pixel_data(row_center_pixel, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
#if RTGI_GUIDE_RESAMPLE_DISTANCE_ONLY
        // Same distance-only gate as the horizontal pass -- see the define's comment in rtgi_guide_resample.inl.
        const float row_center_pixel_width_ws = calc_pixel_width_ws(inv_half_res_size, camera.near_plane, row_center.ndc.z);
        const bool valid = distance(center.position_ws, row_center.position_ws) <= RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD * min(center_pixel_width_ws, row_center_pixel_width_ws);
#else
        // px_threshold loosened to 8x (default 1.2x), same reasoning as the horizontal pass.
        const bool valid = (calc_similar_surface_weight(rcp(center_pixel_width_ws), center.position_ws, center.normal_ws, row_center.position_ws, row_center.normal_ws, RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD)
                           * calc_similar_normal_weight(center.normal_ws, row_center.normal_ws)) > 0.0f;
#endif
        if (!valid)
        {
            continue;
        }

        v_wsum += row_weight.x;
        v_valid_count += row_weight.y;
        // Continue the reservoir: keep this row's already-picked candidate with probability W_row / v_wsum.
        if (rand() * v_wsum < row_weight.x)
        {
            v_held_hit_y = push.attach.h_resample_hit_y.get()[cell];
        }
    }

    const float total_candidates = float(guide_resample_taps * guide_resample_taps);
    const float confidence = 1;//saturate(v_valid_count / total_candidates);

    push.attach.pioneer_guide_hit_y.get()[dtid] = v_held_hit_y;
    push.attach.pioneer_guide_confidence.get()[dtid] = confidence;
}

// Guide debug draws (written by entry_guide_resolve at half-res).
void rtgi_guide_resolve_debug_draw(uint2 dtid, float4 guide_sh_y, float confidence)
{
    let push = rtgi_guide_resolve_push;
    let debug_mode = push.attach.globals.settings.debug_draw_mode;
    const float debug_alpha = 1.0f + push.attach.globals.settings.debug_visualization_blend;
    if (debug_mode == DEBUG_DRAW_MODE_RTGI_GUIDE_CONFIDENCE)
    {
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(confidence), debug_alpha), 2);
    }
    else if (debug_mode == DEBUG_DRAW_MODE_RTGI_GUIDE_DIRECTION)
    {
        // World-space direction as color, darkened by confidence (black == no guiding).
        const float moment_len = length(guide_sh_y.xyz);
        const float3 dir = moment_len > 1e-8f ? guide_sh_y.xyz / moment_len : float3(0.0f, 0.0f, 0.0f);
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4((dir * 0.5f + 0.5f) * confidence, debug_alpha), 2);
    }
}

// === Resolve pass =======================
//
// Preloads the guide-cell neighborhood into groupshared once per workgroup tile (position -- pretransformed
// from depth ONCE per cell, not per output thread -- packed normal, hit position + brightness, confidence),
// the same technique rtgi_pre_filter.hlsl's entry_prepare uses for its own position/normal preload: many
// half-res threads can share the same handful of pioneer cells (up to RTGI_GUIDE_PIONEER_GRID_DIV^2 threads per
// cell), so preloading avoids each of them redundantly re-fetching and re-transforming the same cell's data.
static const int RTGI_GUIDE_RESOLVE_PRELOAD_DIM_X = RTGI_GUIDE_RESAMPLE_X + 2 * RTGI_GUIDE_RESOLVE_EXTENT;
static const int RTGI_GUIDE_RESOLVE_PRELOAD_DIM_Y = RTGI_GUIDE_RESAMPLE_Y + 2 * RTGI_GUIDE_RESOLVE_EXTENT;
// A tile can never touch more distinct pioneer cells than it has pixels, so RTGI_GUIDE_RESAMPLE_X/Y is
// already a safe upper bound on the cell span -- the +2*EXTENT halo on top covers the resolve gather's
// own reach. Comfortably over-sized at RTGI_GUIDE_PIONEER_GRID_DIV == 4 (fewer distinct cells needed
// than that bound allows, just leaves the tile under-filled).
groupshared float4 gs_guide_pos_depth[RTGI_GUIDE_RESOLVE_PRELOAD_DIM_X][RTGI_GUIDE_RESOLVE_PRELOAD_DIM_Y]; // .xyz = ws pos, .w = depth (0 = invalid cell)
groupshared uint   gs_guide_normal_oct[RTGI_GUIDE_RESOLVE_PRELOAD_DIM_X][RTGI_GUIDE_RESOLVE_PRELOAD_DIM_Y];
groupshared float4 gs_guide_hit_y[RTGI_GUIDE_RESOLVE_PRELOAD_DIM_X][RTGI_GUIDE_RESOLVE_PRELOAD_DIM_Y]; // .xyz = candidate's traced hit pos, .w = brightness Y
groupshared float  gs_guide_confidence[RTGI_GUIDE_RESOLVE_PRELOAD_DIM_X][RTGI_GUIDE_RESOLVE_PRELOAD_DIM_Y];

[shader("compute")]
[numthreads(RTGI_GUIDE_RESAMPLE_X, RTGI_GUIDE_RESAMPLE_Y, 1)]
func entry_guide_resolve(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_guide_resolve_push;
    const CameraInfo camera = push.attach.globals.view_camera;
    const uint2 half_res_size = uint2(push.size); // this pass IS dispatched at half-res
    const float2 inv_half_res_size = rcp(float2(half_res_size));

    // Same rotation formula the other two passes use -- MUST stay in sync.
    const uint rotation_frame = push.attach.globals.rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) : 0u;
    const uint2 rotation = uint2(
        rotation_frame % RTGI_GUIDE_PIONEER_GRID_DIV,
        (rotation_frame / RTGI_GUIDE_PIONEER_GRID_DIV) % RTGI_GUIDE_PIONEER_GRID_DIV);
    const int2 pioneer_grid_size = int2((half_res_size + uint2(RTGI_GUIDE_PIONEER_GRID_DIV - 1u, RTGI_GUIDE_PIONEER_GRID_DIV - 1u)) / RTGI_GUIDE_PIONEER_GRID_DIV);

    // Pioneer-cell coordinate of this workgroup tile's origin pixel -- anchor for the preload footprint.
    const int2 group_origin_px   = int2(gid) * int2(RTGI_GUIDE_RESAMPLE_X, RTGI_GUIDE_RESAMPLE_Y);
    const int2 group_origin_cell = (group_origin_px - int2(rotation)) / int(RTGI_GUIDE_PIONEER_GRID_DIV);

    // === GS preload: guide-cell neighborhood (pretransformed position/depth, packed normal, sh_y, confidence) ===
    {
        [unroll]
        for (uint iter_x = 0; iter_x < round_up_div(uint(RTGI_GUIDE_RESOLVE_PRELOAD_DIM_X), RTGI_GUIDE_RESAMPLE_X); ++iter_x)
        {
            [unroll]
            for (uint iter_y = 0; iter_y < round_up_div(uint(RTGI_GUIDE_RESOLVE_PRELOAD_DIM_Y), RTGI_GUIDE_RESAMPLE_Y); ++iter_y)
            {
                const int2 in_idx = int2(iter_x * RTGI_GUIDE_RESAMPLE_X + gtid.x, iter_y * RTGI_GUIDE_RESAMPLE_Y + gtid.y);
                if (all(in_idx < int2(RTGI_GUIDE_RESOLVE_PRELOAD_DIM_X, RTGI_GUIDE_RESOLVE_PRELOAD_DIM_Y)))
                {
                    const int2 cell = clamp(group_origin_cell - RTGI_GUIDE_RESOLVE_EXTENT + in_idx, int2(0, 0), pioneer_grid_size - 1);
                    const uint2 ref_pixel = uint2(cell) * RTGI_GUIDE_PIONEER_GRID_DIV + rotation;
                    const bool ref_valid = all(ref_pixel < half_res_size);
                    const float depth = ref_valid ? push.attach.view_cam_half_res_depth.get()[ref_pixel] : 0.0f;
                    // Pretransform depth -> world position ONCE per shared cell here (like rtgi_pre_filter's
                    // preload), instead of every thread that reads this cell redoing the unprojection.
                    float3 ws_pos = float3(0.0f, 0.0f, 0.0f);
                    if (depth != 0.0f)
                    {
                        ws_pos = pixel_index_to_world_space(camera, float2(ref_pixel * 2u) + 0.5f, depth);
                    }
                    gs_guide_pos_depth[in_idx.x][in_idx.y]  = float4(ws_pos, depth);
                    gs_guide_normal_oct[in_idx.x][in_idx.y] = ref_valid ? push.attach.view_cam_half_res_face_normals.get()[ref_pixel].r : 0u;
                    gs_guide_hit_y[in_idx.x][in_idx.y]      = push.attach.pioneer_guide_hit_y.get()[cell];
                    gs_guide_confidence[in_idx.x][in_idx.y] = push.attach.pioneer_guide_confidence.get()[cell];
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); // gs_guide_* fully written

    if (any(dtid.xy >= half_res_size))
    {
        return;
    }

    const float depth = push.attach.view_cam_half_res_depth.get()[dtid];
    if (depth == 0.0f)
    {
        // Sky: no guiding is meaningful here.
        push.attach.guide_sh_y.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        push.attach.guide_confidence.get()[dtid] = 0.0f;
        return;
    }
    const float3 world_position = pixel_index_to_world_space(camera, float2(dtid * 2u) + 0.5f, depth);
    const float3 face_normal    = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[dtid].r);
    const float receiver_pixel_width_ws = calc_pixel_width_ws(inv_half_res_size, camera.near_plane, depth);

    const int2 own_cell    = (int2(dtid) - int2(rotation)) / int(RTGI_GUIDE_PIONEER_GRID_DIV);
    const int2 local_center = own_cell - group_origin_cell + RTGI_GUIDE_RESOLVE_EXTENT; // index into the GS tile

    // Same gate as the horizontal/vertical passes (see RTGI_GUIDE_RESAMPLE_DISTANCE_ONLY): geometry is a
    // hard cutoff against THIS pixel's own surface, never folded into the weight. Picked via
    // radiance-weighted reservoir sampling, not blended -- averaging multiple cells' reconnected
    // directions together would soften the guide's concentration instead of keeping it a single sharp sample.
    const uint resolve_frame_seed = push.attach.globals.rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) * 257u : 0u;
    rand_seed(dtid.x * 9629u + dtid.y * 10069u + resolve_frame_seed + 43u);

    const float IMPORTANCE_FLOOR = 1e-6f;
    float wsum = 0.0f;
    float4 held_sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  held_confidence = 0.0f;

    [unroll]
    for (int oy = -RTGI_GUIDE_RESOLVE_EXTENT; oy <= RTGI_GUIDE_RESOLVE_EXTENT; ++oy)
    {
        [unroll]
        for (int ox = -RTGI_GUIDE_RESOLVE_EXTENT; ox <= RTGI_GUIDE_RESOLVE_EXTENT; ++ox)
        {
            const int2 gs_idx = local_center + int2(ox, oy);
            const float4 cand_pos_depth = gs_guide_pos_depth[gs_idx.x][gs_idx.y];
            if (cand_pos_depth.w == 0.0f)
            {
                continue; // no valid pioneer reference at this cell (sky/OOB)
            }
#if RTGI_GUIDE_RESAMPLE_DISTANCE_ONLY
            // Same distance-only gate as the horizontal/vertical passes -- see the define's comment in
            // rtgi_guide_resample.inl. cand_pos_depth.w is the candidate cell's raw depth (see the preload
            // above), usable directly as calc_pixel_width_ws's `depth` param.
            const float cand_pixel_width_ws = calc_pixel_width_ws(inv_half_res_size, camera.near_plane, cand_pos_depth.w);
            const bool valid = distance(world_position, cand_pos_depth.xyz) <= RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD * min(receiver_pixel_width_ws, cand_pixel_width_ws);
#else
            const float3 cand_normal = uncompress_normal_octahedral_32(gs_guide_normal_oct[gs_idx.x][gs_idx.y]);
            const bool valid = (calc_similar_surface_weight(rcp(receiver_pixel_width_ws), world_position, face_normal, cand_pos_depth.xyz, cand_normal, RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD)
                               * calc_similar_normal_weight(face_normal, cand_normal)) > 0.0f;
#endif
            if (!valid)
            {
                continue;
            }

            // dist < epsilon guards the degenerate case of the receiver sitting on top of the hit itself;
            // ndotl <= 0 rejects candidates that fall behind this pixel's own hemisphere, which
            // reconnection can produce even when the origin-surface gate above passed (that gate only
            // checked the CANDIDATE CELL's origin surface, never the reconnected direction itself).
            const float4 cand_hit_y = gs_guide_hit_y[gs_idx.x][gs_idx.y]; // .xyz = hit pos, .w = brightness Y
            const float3 to_hit = cand_hit_y.xyz - world_position;
            const float dist = length(to_hit);
            if (dist < 1e-4f)
            {
                continue;
            }
            const float3 recon_dir = to_hit / dist;
            if (dot(face_normal, recon_dir) <= 0.0f)
            {
                continue;
            }

            // Same log1p importance as the horizontal pass -- see its comment. Y itself needs no
            // re-derivation: a Lambertian hit's exitant radiance is the same toward every direction, so
            // the brightness traced at the ORIGINAL pioneer pixel still applies unchanged here.
            const float importance = max(log(cand_hit_y.w + 1.0f), IMPORTANCE_FLOOR);
            wsum += importance;
            if (rand() * wsum < importance)
            {
                // Bake the SH-Y lobe HERE, for THIS receiver, at the actual point of use -- the only SH-Y
                // construction in the whole resample pipeline.
                held_sh_y = y_to_sh(cand_hit_y.w, recon_dir);
                held_confidence = gs_guide_confidence[gs_idx.x][gs_idx.y];
            }
        }
    }

    push.attach.guide_sh_y.get()[dtid]       = held_sh_y;
    push.attach.guide_confidence.get()[dtid] = held_confidence;
    rtgi_guide_resolve_debug_draw(dtid, held_sh_y, held_confidence);
}
