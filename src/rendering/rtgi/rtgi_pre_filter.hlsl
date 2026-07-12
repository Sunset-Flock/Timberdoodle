#pragma once

#include "rtgi_pre_filter.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/debug.glsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPreFilterPush rtgi_pre_filter_prepare_push;

// Outer gather reaches EXTENT cells at STRIDE spacing → EXTENT*STRIDE cells = EXTENT*STRIDE*2 pixels per side.
static const int QUAD_HALO_CELLS   = RTGI_QUAD_FILTER_EXTENT * RTGI_QUAD_FILTER_STRIDE;
static const int TOTAL_FILTER_REACH = QUAD_HALO_CELLS * 2;

static const int PRELOAD_SIZE_X = RTGI_PRE_BLUR_PREPARE_X + TOTAL_FILTER_REACH * 2;
static const int PRELOAD_SIZE_Y = RTGI_PRE_BLUR_PREPARE_Y + TOTAL_FILTER_REACH * 2;

// idx is a groupshared *tile* index (same space as gs_quad_pos_depth / gs_quad_normals_oct).
// Normals are preloaded into LDS in the position-preload loop, so this is a plain LDS read.
#define LOAD_NORMAL_OCT(tile_idx) gs_quad_normals_oct[(tile_idx).x][(tile_idx).y]

// --- 2x2-blurred radiance grid and per-quad CoV ---
// gs_radiance_blurred covers (WG/2 + 4) x (WG/2 + 4) = 12x12 entries.
// Each entry is the average Y (radiance) of a 2x2 block of the per-pixel ray means, stride 2.
// Entries 0–1 and 10–11 extend 2 blur-cells (= 4 input pixels) past each workgroup edge.
static const int NUM_QUADS_X      = RTGI_PRE_BLUR_PREPARE_X / 2;
static const int NUM_QUADS_Y      = RTGI_PRE_BLUR_PREPARE_Y / 2;
static const int GS_RADIANCE_DIM_X   = NUM_QUADS_X + RTGI_QUAD_FILTER_EXTENT * RTGI_QUAD_FILTER_STRIDE * 2;
static const int GS_RADIANCE_DIM_Y   = NUM_QUADS_Y + RTGI_QUAD_FILTER_EXTENT * RTGI_QUAD_FILTER_STRIDE * 2;
static const int GS_RADIANCE_TOTAL   = GS_RADIANCE_DIM_X * GS_RADIANCE_DIM_Y;
// Surrounding cells of a (2*E+1)^2 neighborhood minus center, split evenly across 4 lanes.
// Works for E=1 (8→2/lane), E=2 (24→6/lane), E=3 (48→12/lane).
static const int QUAD_FILTER_DIM   = RTGI_QUAD_FILTER_EXTENT * 2 + 1;
groupshared float4 gs_blur_color[GS_RADIANCE_DIM_X][GS_RADIANCE_DIM_Y];     // .xyz = rgb, .w = radiance (geometry-aware averaged, perceptual space)
groupshared float4 gs_quad_position_depth[GS_RADIANCE_DIM_X][GS_RADIANCE_DIM_Y];        // .xyz = representative vs pos, .w = depth (sky→0)
groupshared float3 gs_quad_normal_ws[GS_RADIANCE_DIM_X][GS_RADIANCE_DIM_Y]; // unpacked face normal per blur-cell (precomputed so the quad loop skips the octahedral decode)
groupshared float  gs_quad_rayshortness[GS_RADIANCE_DIM_X][GS_RADIANCE_DIM_Y]; // mean ray shortness [0,1] per blur-cell
groupshared float  gs_quad_raycount[GS_RADIANCE_DIM_X][GS_RADIANCE_DIM_Y];     // mean rays-shot per blur-cell (for firefly-ring ceiling reduction)
groupshared float4 gs_quad_pos_depth[PRELOAD_SIZE_X][PRELOAD_SIZE_Y];  // .xyz = ws pos, .w = depth; full 12×12 tile
groupshared uint   gs_quad_normals_oct[PRELOAD_SIZE_X][PRELOAD_SIZE_Y]; // packed octahedral normals; full tile, indexed by tile coord


[shader("compute")]
[numthreads(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y,1)]
func entry_prepare(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_pre_filter_prepare_push;
    RWTexture2D<float4> dbg = push.attach.debug_image.get();

    // Load and precalculate constants
    CameraInfo *camera = &push.attach.globals->view_camera;
    const float2 inv_half_res_render_target_size = rcp(float2(push.size));
    const uint2 clamped_index = min( dtid.xy, push.size - 1u );      // Can not early out because we perform group shared memory barriers later!

    // Center pixel sits at gtid + TOTAL_FILTER_REACH in the preload tile (maps src texel → tile index)
    const int2 grad_idx = int2(gtid) + TOTAL_FILTER_REACH;

    // Load Pixel Data
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];

    const float pixel_width_ws = calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, depth);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    // === GS position preload: all 64 threads reconstruct WS positions for the full 16x16 area ===
    {
        [unroll]
        for (uint iter_x = 0; iter_x < round_up_div(PRELOAD_SIZE_X, RTGI_PRE_BLUR_PREPARE_X); ++iter_x)
        {
            [unroll]
            for (uint iter_y = 0; iter_y < round_up_div(PRELOAD_SIZE_Y, RTGI_PRE_BLUR_PREPARE_Y); ++iter_y)
            {
                const int2 in_idx = int2(iter_x * RTGI_PRE_BLUR_PREPARE_X + gtid.x, iter_y * RTGI_PRE_BLUR_PREPARE_Y + gtid.y);
                if (all(in_idx < int2(PRELOAD_SIZE_X, PRELOAD_SIZE_Y)))
                {
                    const int2 src = clamp(int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) - TOTAL_FILTER_REACH + in_idx, int2(0, 0), int2(push.size) - 1);
                    // No-ray pixels are folded into the sky sentinel (see main preload below) so the quad
                    // reduction / star blur skip them too.
                    const float src_ray_count = float(push.attach.ray_count_image.get()[src]);
                    const float pd = (src_ray_count == 0.0f) ? 0.0f : push.attach.view_cam_half_res_depth.get()[src];
                    const float2 src_uv_gs = (float2(src) + 0.5f) * inv_half_res_render_target_size;
                    const float4 pp = mul(camera->inv_view_proj, float4(src_uv_gs * 2.0f - 1.0f, pd, 1.0f));
                    gs_quad_pos_depth[in_idx.x][in_idx.y] = float4(pd != 0.0f ? pp.xyz / pp.w : float3(0, 0, 0), pd);
                    gs_quad_normals_oct[in_idx.x][in_idx.y] = push.attach.view_cam_half_res_normals.get()[src];
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); // gs_quad_pos_depth + gs_quad_normals_oct fully written
    const float3 world_position    = gs_quad_pos_depth[grad_idx.x][grad_idx.y].xyz;
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(LOAD_NORMAL_OCT(grad_idx));

    // === Preload 2x2-blurred radiance/rgb into gs_blur_color (geometry-aware) ===
    // One thread per blur-cell (144 entries ≤ 256-thread WG).
    // For each 2×2 block: pick the pixel whose depth is closest to the pseudo-median (robust
    // representative), then only average radiance from pixels that lie on the same surface
    // (planar distance < 1 pixel). Store the representative vs position and normal so the
    // per-quad CoV filter can do consistent geometry tests without per-lane divergence.
    {
        const uint flat_i = gtid.y * RTGI_PRE_BLUR_PREPARE_X + gtid.x;
        if (flat_i < uint(GS_RADIANCE_TOTAL))
        {
            const int bx = int(flat_i) % GS_RADIANCE_DIM_X;
            const int by = int(flat_i) / GS_RADIANCE_DIM_X;
            const int2 src_base = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) + int2(bx - QUAD_HALO_CELLS, by - QUAD_HALO_CELLS) * 2;

            // Load all 4 pixels including normals upfront — needed for normal-aware rep selection.
            float  depths[4];
            float3 ws_pos[4];
            float3 perceptual_rgbs[4];
            uint   normals_oct[4];
            float3 normals_ws[4];
            float  ray_shortness[4]; // ray-length texture now stores mean shortness [0,1] directly
            float  ray_counts[4];    // rays shot per pixel this frame
            [unroll]
            for (int pi = 0; pi < 4; pi++)
            {
                const int2 src_idx = clamp(src_base + int2(pi & 1, pi >> 1), int2(0, 0), int2(push.size) - 1);
                const int2 gs_pi = int2(bx * 2 + (pi & 1), by * 2 + (pi >> 1));
                const float4 gs_pd = gs_quad_pos_depth[gs_pi.x][gs_pi.y];
                const float pd = gs_pd.w;
                depths[pi] = pd;
                ws_pos[pi] = gs_pd.xyz;
                normals_oct[pi] = LOAD_NORMAL_OCT(gs_pi);
                normals_ws[pi]  = uncompress_normal_octahedral_32(normals_oct[pi]);
                ray_counts[pi]    = float(push.attach.ray_count_image.get()[src_idx]);
                // .rgb = geometric mean of the pixel's rays in log space; .a = mean ray shortness [0,1].
                // Perceptual radiance is inferred from the log rgb (no dedicated radiance channel).
                const float4 perceptual_geo = push.attach.perceptual_rgb_shortness.get()[src_idx];
                perceptual_rgbs[pi]      = perceptual_geo.rgb;
                ray_shortness[pi]        = perceptual_geo.a;
            }

            int rep_i = quad_downsample_representative(depths);

            const float3 rep_normal_ws = normals_ws[rep_i];

            // Accumulate in perceptual (log) space so the blur-cell stores the geometric mean,
            // not the arithmetic mean — avoids Jensen's inequality bias in firefly ceiling and CoV.
            float3 rgb_acc     = float3(0.0f, 0.0f, 0.0f);
            float  radiance_weight = 0.0f;
            float  rs_acc      = 0.0f;
            float  rc_acc      = 0.0f;
            float  diffuse_weight = 0.0f;
            if (depths[rep_i] != 0.0f)
            {
                [unroll]
                for (int pi = 0; pi < 4; pi++)
                {
                    const float gw = depths[pi] != 0.0f ? calc_similar_surface_weight(pixel_width_ws_rcp, ws_pos[rep_i], rep_normal_ws, ws_pos[pi], normals_ws[pi], 4.0f) : 0.0f;
                    rgb_acc  += perceptual_rgbs[pi]  * gw;
                    radiance_weight += gw;
                    rs_acc += ray_shortness[pi] * gw;
                    rc_acc += ray_counts[pi] * gw;
                    diffuse_weight += gw;
                }
            }
            const float3 blur_perceptual_rgb  = radiance_weight > 0.0f ? rgb_acc / radiance_weight : perceptual_rgbs[rep_i];
            // Perceptual radiance is derived from the (geometry-averaged) perceptual rgb, no separate accumulator.
            const float  blur_perceptual_radiance = perceptual_radiance_from_rgb(blur_perceptual_rgb);
            gs_blur_color[bx][by]           = float4(blur_perceptual_rgb, blur_perceptual_radiance);
            gs_quad_position_depth[bx][by]  = float4(ws_pos[rep_i], depths[rep_i]);
            gs_quad_normal_ws[bx][by]       = rep_normal_ws;
            gs_quad_rayshortness[bx][by]    = radiance_weight > 0.0f ? rs_acc / radiance_weight : 0.0f;
            gs_quad_raycount[bx][by]        = radiance_weight > 0.0f ? rc_acc / radiance_weight : 1.0f;
        }
    }

    GroupMemoryBarrierWithGroupSync(); // gs_blur_color, gs_quad_position_depth, gs_quad_normal_ws, gs_quad_rayshortness all visible

    // === Per-pixel quad filter ===
    // Each pixel independently tests the (2*EXTENT+1)^2 surrounding quads against its own
    // position/normal. No cooperative partial-sum stage — one barrier instead of two.
    // Pixels on surface discontinuities within a 2x2 quad now correctly weight neighbours
    // against their own surface rather than the quad representative's surface.
    const int2 pixel_quad_id   = int2(gtid) / 2;
    const int2 pixel_gs_center = pixel_quad_id + int2(QUAD_HALO_CELLS, QUAD_HALO_CELLS);

    float3 perceptual_rgb_acc  = float3(0.0f, 0.0f, 0.0f); // geometry-weighted mean log rgb; radiance derived from it
    float geo_weight_acc    = 0.0f;
    // Geometry-weighted rays-shot summed over the RING taps (center taps are added separately via
    // center_quad_ray_sum_acc). Feeds neighborhood_sample_fitness: a neighborhood with many rays raises
    // the fitness even when few taps pass the geometry test.
    float ring_raycount_geo_acc = 0.0f;
    float ray_shortness_acc = 0.0f;
    // Firefly reference (ceiling) mean — built ONLY from the surrounding quad blur-cells, never from
    // the center quad. The center pixels are the ones being clamped, so including them would let a
    // firefly inflate its own ceiling and pass through (worst at low extent where the surround is small).
    float3 ff_perceptual_rgb_sum   = float3(0.0f, 0.0f, 0.0f);
    float ff_weight         = 0.0f;
    float ff_ray_count_sum  = 0.0f; // rays-shot summed over the firefly ring (same taps as ff_*)
    // Geometry-AWARE firefly reference: same ring taps but weighted by surface similarity (gw), so only
    // neighbors on the same surface contribute to the ceiling. This is the primary reference; the
    // unweighted ff_* mean above is only used as a fallback when the footprint drops too low.
    float3 ff_perceptual_rgb_sum_geo = float3(0.0f, 0.0f, 0.0f);
    float ff_weight_geo = 0.0f;
    // Surrounding quads — skip center (qx==0, qy==0), handled at pixel resolution below.
    [unroll]
    for (int qy = -RTGI_QUAD_FILTER_EXTENT; qy <= RTGI_QUAD_FILTER_EXTENT; qy++)
    {
        [unroll]
        for (int qx = -RTGI_QUAD_FILTER_EXTENT; qx <= RTGI_QUAD_FILTER_EXTENT; qx++)
        {
            if (qx == 0 && qy == 0) continue;
            const int2   sq         = pixel_gs_center + int2(qx, qy) * RTGI_QUAD_FILTER_STRIDE;
            const float4 sq_vs_data = gs_quad_position_depth[sq.x][sq.y];
            const float  sq_depth   = sq_vs_data.w;
            const bool   sq_sky     = sq_depth == 0.0f;
            const float3 sq_normal_ws = gs_quad_normal_ws[sq.x][sq.y];
            const float  gw = !sq_sky ? calc_similar_surface_weight( pixel_width_ws_rcp, world_position, pixel_face_normal, sq_vs_data.xyz, sq_normal_ws, 4.0f) : 0.0f;

            // gs_blur_color is already in log (perceptual) space.
            const float3 perceptual_rgb  = gs_blur_color[sq.x][sq.y].xyz;
            perceptual_rgb_acc   += perceptual_rgb  * gw;
            geo_weight_acc    += gw;
            ring_raycount_geo_acc += gs_quad_raycount[sq.x][sq.y] * gw;
            ray_shortness_acc += gs_quad_rayshortness[sq.x][sq.y] * gw;

            if (!sq_sky)
            {
                ff_ray_count_sum += gs_quad_raycount[sq.x][sq.y];
                ff_weight += 1.0f;
                // Geometry-aware reference (primary): weight by surface similarity to the center pixel.
                ff_perceptual_rgb_sum_geo  += perceptual_rgb * gw;
                ff_weight_geo += gw;
                // Non-geometry-aware reference: unweighted over all non-sky ring neighbors (ignores surface
                // similarity). Genuinely distinct from the geo-aware mean; used as the fallback.
                ff_perceptual_rgb_sum  += perceptual_rgb;
            }
        }
    }

    // Center quad: 4 individual pixel taps at full (non-quad) resolution.
    // The quad representative may disagree with edge pixels inside the quad — testing the
    // 4 raw pixels eliminates false inf sentinels at depth discontinuities.
    // Apply linear_to_perceptual here (raw pixels, unlike gs_blur_color which pre-logs).
    float center_weight_acc = 0.0f;
    float center_quad_ray_sum_acc = 0.0f; // geometry-weighted sum of the quad pixels' ray counts
    {
        const int2  center_quad_base = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) + pixel_quad_id * 2;
        [unroll]
        for (int pi = 0; pi < 4; pi++)
        {
            const int2   tap_idx   = clamp(center_quad_base + int2(pi & 1, pi >> 1), int2(0, 0), int2(push.size) - 1);
            const int2   tap_gs    = int2(pixel_quad_id.x * 2 + (pi & 1), pixel_quad_id.y * 2 + (pi >> 1)) + TOTAL_FILTER_REACH;
            const float4 tap_pd    = gs_quad_pos_depth[tap_gs.x][tap_gs.y];
            const float  tap_depth = tap_pd.w;
            if (tap_depth == 0.0f) continue;
            const float3 tap_pos_ws = tap_pd.xyz;
            const float3 tap_nrm_ws  = uncompress_normal_octahedral_32(LOAD_NORMAL_OCT(tap_gs));
            const float  tap_gw      = calc_similar_surface_weight(pixel_width_ws_rcp, world_position, pixel_face_normal, tap_pos_ws, tap_nrm_ws, 4.0f);
            // .rgb = geometric mean of the tap pixel's rays in log space; .a = mean ray shortness [0,1].
            const float4 tap_perceptual_geo = push.attach.perceptual_rgb_shortness.get()[tap_idx];
            const float3 tap_perceptual_rgb = tap_perceptual_geo.rgb;
            const float  tap_close   = tap_perceptual_geo.a;

            perceptual_rgb_acc   += tap_perceptual_rgb  * tap_gw;
            geo_weight_acc    += tap_gw;
            ray_shortness_acc += tap_close    * tap_gw;
            // NOTE: center-quad pixels are deliberately NOT added to the firefly reference
            // (ff_*). The ceiling must be a mean of the *surrounding* neighborhood only — a
            // pixel must never contribute to the ceiling used to clamp itself.
            center_weight_acc += tap_gw;
            center_quad_ray_sum_acc += float(push.attach.ray_count_image.get()[tap_idx]) * tap_gw;
        }
    }

    const float3 mean_perceptual_rgb = geo_weight_acc > 0.0f ? perceptual_rgb_acc / geo_weight_acc : float3(0.0f, 0.0f, 0.0f);
    const float3 geo_perceptual_rgb    = ff_weight_geo > 0.0f ? ff_perceptual_rgb_sum_geo / ff_weight_geo : mean_perceptual_rgb;
    const float3 nongeo_perceptual_rgb = ff_weight     > 0.0f ? ff_perceptual_rgb_sum     / ff_weight     : mean_perceptual_rgb;
    const float3 fallback_perceptual_rgb = min(geo_perceptual_rgb, nongeo_perceptual_rgb);
    const float3 firefly_perceptual_rgb_mean_geo = ff_weight_geo > 0.0f ? geo_perceptual_rgb    : fallback_perceptual_rgb;
    const float3 firefly_perceptual_rgb_mean     = ff_weight     > 0.0f ? nongeo_perceptual_rgb : fallback_perceptual_rgb;

    const float  ray_shortness_mean = geo_weight_acc > 0.0f ? ray_shortness_acc / geo_weight_acc : 0.0f;
    const bool   pixel_matches_quad = center_weight_acc > 0.0f;
    const float  total_quad_taps = float(QUAD_FILTER_DIM * QUAD_FILTER_DIM - 1 + 4);  // surrounding quads + 4 center pixels
    const float  neighborhood_raycount_geo = ring_raycount_geo_acc + center_quad_ray_sum_acc;
    const float  neighborhood_sample_fitness = min(neighborhood_raycount_geo, total_quad_taps) / total_quad_taps;
    const float  neighborhood_sample_fitness_sharp = min(neighborhood_sample_fitness, 0.25f) * 4.0f;

    let rtgi = push.attach.globals->rtgi_settings;

    // === Firefly Filter ===
    float4 filtered_diffuse;
    float2 filtered_diffuse2;
    float firefly_energy_factor = 1.0f;
    {
        const float EPSILON = 1e-8f;
        // Firefly ceiling reference: geometry-aware neighborhood mean, tightened with the non-geo-aware mean
        // where the sample fitness is low so a sparse (outlier-dominated) mean can't inflate the ceiling.
        float3 perceptual_rgb_for_ceiling = firefly_perceptual_rgb_mean_geo;
        if (neighborhood_sample_fitness_sharp < 1.0f)
        {
            perceptual_rgb_for_ceiling = min(perceptual_rgb_for_ceiling, firefly_perceptual_rgb_mean);
        }
        // Perceptual tolerance: how far (in perceptual / log space) above the neighborhood mean a ray may
        // reach before it is clamped. It is ADDED to the mean's perceptual rgb and converted back to
        // linear. Raised by the neighborhood sample fitness (well-sampled neighborhoods admit more energy),
        // and tightened when the pixel does not match its quad.
        // Per channel we also SUBTRACT RTGI_CHANNEL_PERCEIVED_BRIGHTNESS_LN (<= 0, 0 for green): red and
        // blue are perceived dimmer than green, so this gives their ceilings extra headroom — a channel
        // needs proportionally more linear energy to look as bright.
        const float perceptual_tolerance = rtgi.firefly_perceptual_tolerance * (pixel_matches_quad ? 1.0f : 0.1f);
        const float3 ceil_rgb = perceptual_to_linear(
            perceptual_rgb_for_ceiling + perceptual_tolerance);
        const float ceil_merged = dot(ceil_rgb,       RTGI_CHANNEL_PERCEIVED_BRIGHTNESS);

        // Blend the pixel's rays directly from the ray list (directional SH radiance + CoCg averaged over the
        // rays, exactly like the old blend pass), hue-preservingly firefly-clamping EACH ray to the ceiling
        // first when the filter is enabled. This is the SOLE source of the pixel's filtered diffuse, so the
        // blend pass no longer outputs diffuse/diffuse2.
        const uint  ray_offset = push.attach.pixel_ray_alloc.get()[clamped_index];
        const uint  ray_count  = push.attach.ray_count_image.get()[clamped_index];
        const bool  do_clamp   = rtgi.firefly_filter_enabled != 0;
        const bool  ff_mono    = rtgi.firefly_clamp_mode == 1;
        float4 acc_sh   = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 acc_cocg = float2(0.0f, 0.0f);
        float  energy_pre  = 0.0f;
        float  energy_post = 0.0f;
        if (ray_count > 0u)
        {
            const float inv_count = 1.0f / float(ray_count);
            for (uint s = 0u; s < ray_count; ++s)
            {
                const RtgiRayResult res = push.attach.ray_result[ray_offset + s];
                const float3 ray_rgb = res.radiance;
                float3 clamped = ray_rgb;
                if (do_clamp)
                {
                    const float ray_merged  = dot(ray_rgb, RTGI_CHANNEL_PERCEIVED_BRIGHTNESS);
                    // Both modes scale ALL channels by one over-exposure ratio (hue-preserving); they only
                    // differ in how that ratio is derived from the per-channel ceilings:
                    //   monochrome: merge ray + ceiling into a single perceived-brightness scalar, then take
                    //               the ratio of the merged values (a single bright channel is diluted by
                    //               the merge, so it clamps less).
                    //   rgb:        take the per-channel over-exposure ratios and keep the most conservative
                    //               (largest), so the single worst channel decides the clamp.
                    float over;
                    if (ff_mono)
                    {
                        over = (ray_merged / ceil_merged);
                    }
                    else
                    {
                        const float3 ex = (ray_rgb * RTGI_CHANNEL_PERCEIVED_BRIGHTNESS) / ceil_rgb;
                        over = max(ex.r, max(ex.g, ex.b));
                    }
                    clamped = ray_rgb / max(over, 1.0f);
                }
                const float3 dir = uncompress_normal_octahedral_32(res.packed_dir);
                float4 sh_y; float2 cocg;
                radiance_to_y_co_cg_sh(clamped, dir, sh_y, cocg);
                acc_sh   += sh_y  * inv_count;
                acc_cocg += cocg  * inv_count;
                energy_pre  += dot(ray_rgb, float3(1.0f, 1.0f, 1.0f)) * inv_count;
                energy_post += dot(clamped, float3(1.0f, 1.0f, 1.0f)) * inv_count;
            }
        }
        filtered_diffuse  = acc_sh;
        filtered_diffuse2 = acc_cocg;
        if (do_clamp && rtgi.pre_blur_firefly_energy_compensation_enabled != 0)
        {
            firefly_energy_factor = energy_pre / max(energy_post, 1e-6f);
        }
    }

    // --- Ambient occlusion guide ---
    const float ao_guide = (1.0f - sqrt(ray_shortness_mean));

    // Monochrome (perceptual) radiance derived straight from the perceptual rgb mean — no linear round-trip.
    const float radiance_mean_perceptual = geo_weight_acc > 0.0f
        ? perceptual_radiance_from_rgb(mean_perceptual_rgb)
        : linear_to_perceptual(0.0f, push.attach.globals.inv_exposure);

    push.attach.pre_filtered_diffuse_image.get()[dtid] = filtered_diffuse;
    push.attach.pre_filtered_diffuse2_image.get()[dtid] = filtered_diffuse2;
    push.attach.firefly_factor_image.get()[dtid] = firefly_energy_factor;
    push.attach.perceptual_radiance_image.get()[dtid] = radiance_mean_perceptual;
    push.attach.ao_guide_image.get()[dtid] = ao_guide;

    const float debug_alpha = 1.0f + push.attach.globals.settings.debug_visualization_blend;
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_AO_GUIDE)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(ao_guide), debug_alpha), 2);
    }
    else if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_PERCEPTUAL_MEAN)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(perceptual_radiance_colormap(radiance_mean_perceptual, push.attach.globals.exposure), debug_alpha + 1.0f), 2);
    }
}
