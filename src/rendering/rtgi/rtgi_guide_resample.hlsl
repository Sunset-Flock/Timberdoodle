#pragma once

#include "rtgi_guide_resample.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/debug.glsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiGuideResampleHorizontalPush rtgi_guide_resample_horizontal_push;
[[vk::push_constant]] RtgiGuideResampleVerticalPush rtgi_guide_resample_vertical_push;

// === RTGI Guide Resample =========================================================================
// Turns the sparse pioneer trace (pioneer_hit_y, one ray per pioneer cell -- see pioneer_ray_gen in
// rtgi_trace_diffuse.hlsl) into one RIS-picked (hit position, brightness) per pioneer cell
// (pioneer_guide_hit_y), via two passes: horizontal resample, vertical resample. The trace reads those picks per
// ray (rtgi_fetch_ray_guide in rtgi_trace_diffuse.hlsl) and does the reconnection there.
//
// RECONNECTION, NOT A BAKED DIRECTION
// Every candidate carried through the horizontal and vertical passes is a (hit position X, brightness
// Y) pair, never a direction. The pioneer ray that found X was cast from ITS OWN pixel, generally a
// different surface point than whichever half-res pixel eventually asks for a guide -- baking that
// ray's direction in early would only be correct for the one pixel it was cast from. Y needs no such
// caveat: the traced surface is Lambertian (this whole pipeline is diffuse-only GI), so its exitant
// radiance is the SAME toward every direction, and stays valid for any receiver. So the horizontal and
// vertical passes pick candidates purely by position-gated brightness -- they never look at or need a
// direction at all. Only the trace (rtgi_fetch_ray_guide) ever turns a winning (X, Y) into a real
// direction: normalize(X - receiver_position), computed fresh for whichever ray is actually asking. That's the reconnection -- the direction is re-derived per receiver instead of copied from
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
    const float center_pixel_width_ws = rtgi_half_res_pixel_width_ws(inv_half_res_size, camera.near_plane, center.depth_vs);

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
        const float candidate_pixel_width_ws = rtgi_half_res_pixel_width_ws(inv_half_res_size, camera.near_plane, candidate.depth_vs);
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
        return;
    }
    const PixelData center = calc_pixel_data(center_pixel, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
    const float center_pixel_width_ws = rtgi_half_res_pixel_width_ws(inv_half_res_size, camera.near_plane, center.depth_vs);

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
        const float row_center_pixel_width_ws = rtgi_half_res_pixel_width_ws(inv_half_res_size, camera.near_plane, row_center.depth_vs);
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

    push.attach.pioneer_guide_hit_y.get()[dtid] = v_held_hit_y;

    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_GUIDE_LOW_RES_DIRECTION)
    {
        // This cell's pick as the direction from its reference pixel to the picked hit, drawn over every half-res
        // pixel the cell covers (the pixels whose own cell this is), so the pioneer resolution shows as blocks.
        // Black = no pick.
        const float3 to_hit = v_held_hit_y.xyz - center.position_ws;
        const float dist = length(to_hit);
        const float3 color = (v_held_hit_y.w > 0.0f && dist > 1e-4f) ? (to_hit / dist) * 0.5f + 0.5f : float3(0.0f, 0.0f, 0.0f);
        const float debug_alpha = 1.0f + push.attach.globals.settings.debug_visualization_blend;
        for (uint by = 0u; by < RTGI_GUIDE_PIONEER_GRID_DIV; ++by)
        {
            for (uint bx = 0u; bx < RTGI_GUIDE_PIONEER_GRID_DIV; ++bx)
            {
                const uint2 px = dtid * RTGI_GUIDE_PIONEER_GRID_DIV + rotation + uint2(bx, by);
                if (all(px < half_res_size))
                {
                    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, px, float4(color, debug_alpha), 2);
                }
            }
        }
    }
}
