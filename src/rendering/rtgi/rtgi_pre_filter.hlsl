#pragma once

#include "rtgi_pre_filter.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/debug.glsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPreFilterPush rtgi_pre_filter_prepare_push;

// Set to 1 to use the GS quad path for all pre-filter computations:
// guide mean, ray-length guide, star blur, and footprint quality.
// Set to 0 to use the legacy per-pixel paths. (Controlled by RTGI_USE_QUAD in rtgi_shared.hlsl)

// Because of the +1
static const float PERCEPTUAL_SPACE_MULTIPLIER = 1e1f;
static const float MAX_VISIBILITY_PIXEL_RANGE  = 48.0f;

func linear_to_perceptual(float v) -> float
{
    return log(max(v, 1e-8f));
}
__generic<uint N>
func linear_to_perceptual(vector<float, N> v) -> vector<float, N>
{
    return log(max(v, 1e-8f));
}

func perceptual_to_linear(float v) -> float
{
    return exp(v);
}
__generic<uint N>
func perceptual_to_linear(vector<float, N> v) -> vector<float, N>
{
    return (exp(v));
}

// The center blur is used to preserve firefly energy by flat filtering all pixels in a star kernel covering 5 pixels.
func is_part_of_center_blur(int2 index) -> bool
{
    return (abs(index.x) + abs(index.y) <= 1);
}

static const int FILTER_STRIDE = 1;
#if RTGI_USE_QUAD
// Outer gather reaches EXTENT cells at STRIDE spacing → EXTENT*STRIDE cells = EXTENT*STRIDE*2 pixels per side.
static const int QUAD_HALO_CELLS   = RTGI_QUAD_FILTER_EXTENT * RTGI_QUAD_FILTER_STRIDE;
static const int TOTAL_FILTER_REACH = QUAD_HALO_CELLS * 2;
#else
static const int TOTAL_FILTER_REACH = 2;
#endif
static const int TOTAL_INNER_FILTER_REACH = 1;                                         // Inner filter calculates geometry aware metrics: footprint valid samples ratio, star blur, ray length average
static const int FILTER_TAPS_TOTAL = (TOTAL_FILTER_REACH * 2 + 1) * (TOTAL_FILTER_REACH * 2 + 1);
static const int INNER_FILTER_TAPS_TOTAL = (TOTAL_INNER_FILTER_REACH * 2 + 1) * (TOTAL_INNER_FILTER_REACH * 2 + 1);
static const int STAR_BLUR_TAPS = 5;                                                    // 5 generally yields the best cost to use ratio.
static const int GEOMETRIC_MEAN_TAPS_TOTAL = FILTER_TAPS_TOTAL - STAR_BLUR_TAPS;              // 20 geometric mean taps are a good ratio against the 5 center blur taps.

static const int PRELOAD_SIZE_X = RTGI_PRE_BLUR_PREPARE_X + TOTAL_FILTER_REACH * 2;
static const int PRELOAD_SIZE_Y = RTGI_PRE_BLUR_PREPARE_Y + TOTAL_FILTER_REACH * 2;
#if !RTGI_USE_QUAD
groupshared float4 gs_sample_positions[PRELOAD_SIZE_X][PRELOAD_SIZE_Y]; // .xyz = sample world-space pos, .w = depth
// Full PRELOAD_SIZE (20x20) gives a 2-pixel border, supporting stride 1 and stride 2 gradients
// RGB packed as R8G8B8 in the low 24 bits of a uint
groupshared uint gs_albedo_rgb[PRELOAD_SIZE_X][PRELOAD_SIZE_Y];
groupshared float3 gs_sample_normal_ws_nonquad[PRELOAD_SIZE_X][PRELOAD_SIZE_Y];
#endif

// idx is a groupshared *tile* index (same space as gs_quad_pos_depth / gs_quad_normals_oct).
// Normals are preloaded into LDS in the position-preload loop, so this is a plain LDS read.
#define LOAD_NORMAL_OCT(tile_idx) gs_quad_normals_oct[(tile_idx).x][(tile_idx).y]

// --- 2x2-blurred luma grid and per-quad CoV ---
// gs_luma_blurred covers (WG/2 + 4) x (WG/2 + 4) = 12x12 entries.
// Each entry is the average Y (luma) of a 2x2 block of diffuse_raw, stride 2.
// Entries 0–1 and 10–11 extend 2 blur-cells (= 4 input pixels) past each workgroup edge.
static const int NUM_QUADS_X      = RTGI_PRE_BLUR_PREPARE_X / 2;
static const int NUM_QUADS_Y      = RTGI_PRE_BLUR_PREPARE_Y / 2;
static const int NUM_QUADS        = NUM_QUADS_X * NUM_QUADS_Y;
static const int GS_LUMA_DIM_X   = NUM_QUADS_X + RTGI_QUAD_FILTER_EXTENT * RTGI_QUAD_FILTER_STRIDE * 2;
static const int GS_LUMA_DIM_Y   = NUM_QUADS_Y + RTGI_QUAD_FILTER_EXTENT * RTGI_QUAD_FILTER_STRIDE * 2;
static const int GS_LUMA_TOTAL   = GS_LUMA_DIM_X * GS_LUMA_DIM_Y;
// Surrounding cells of a (2*E+1)^2 neighborhood minus center, split evenly across 4 lanes.
// Works for E=1 (8→2/lane), E=2 (24→6/lane), E=3 (48→12/lane).
static const int QUAD_FILTER_DIM   = RTGI_QUAD_FILTER_EXTENT * 2 + 1;
groupshared float4 gs_blur_color[GS_LUMA_DIM_X][GS_LUMA_DIM_Y];     // .xyz = rgb, .w = luma (geometry-aware averaged, perceptual space)
groupshared float4 gs_quad_position_depth[GS_LUMA_DIM_X][GS_LUMA_DIM_Y];        // .xyz = representative vs pos, .w = depth (sky→0)
groupshared uint   gs_quad_normal_oct[GS_LUMA_DIM_X][GS_LUMA_DIM_Y]; // packed octahedral face normal per blur-cell
#if RTGI_USE_QUAD
groupshared float  gs_quad_rayshortness[GS_LUMA_DIM_X][GS_LUMA_DIM_Y]; // mean ray shortness [0,1] per blur-cell
groupshared float4 gs_quad_pos_depth[PRELOAD_SIZE_X][PRELOAD_SIZE_Y];  // .xyz = ws pos, .w = depth; full 12×12 tile
groupshared uint   gs_quad_normals_oct[PRELOAD_SIZE_X][PRELOAD_SIZE_Y]; // packed octahedral normals; full tile, indexed by tile coord
#endif

// Converts a surround flat index [0, NUM_SURROUND) to a (dx,dy) offset, skipping center (0,0).
// Row-major over (2*E+1)x(2*E+1), center at flat index E*(2*E+1)+E.
func surround_offset(int global_k) -> int2
{
    static const int center_flat = RTGI_QUAD_FILTER_EXTENT * QUAD_FILTER_DIM + RTGI_QUAD_FILTER_EXTENT;
    const int actual_k = global_k < center_flat ? global_k : global_k + 1;
    return int2(actual_k % QUAD_FILTER_DIM - RTGI_QUAD_FILTER_EXTENT,
                actual_k / QUAD_FILTER_DIM - RTGI_QUAD_FILTER_EXTENT);
}

func src_to_preload_index(int2 src_index, int2 gid) -> int2
{
    const int2 src_group_base_index = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) - TOTAL_FILTER_REACH;
    return clamp(src_index - src_group_base_index, int2(0,0), int2(PRELOAD_SIZE_X-1, PRELOAD_SIZE_Y-1));
}

func unpack_rgb(uint p) -> float3 { return float3(p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF) * (1.0f / 255.0f); }


[shader("compute")]
[numthreads(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y,1)]
func entry_prepare(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_pre_filter_prepare_push;
    RWTexture2D<float4> dbg = push.attach.debug_image.get();

    // Load and precalculate constants
    CameraInfo *camera = &push.attach.globals->view_camera;
    const float2 inv_half_res_render_target_size = rcp(float2(push.size));
    const uint2 half_res_index = dtid.xy;
    const uint2 clamped_index = min( half_res_index, push.size - 1u );      // Can not early out because we perform group shared memory barriers later!



    // Center pixel sits at gtid + TOTAL_FILTER_REACH in the preload tile (maps src texel → tile index)
    const int2 grad_idx = int2(gtid) + TOTAL_FILTER_REACH;

    // Load Pixel Data
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];

    // const float pixel_samplecnt = push.attach.half_res_samplecnt.get()[clamped_index];

    const float pixel_vs_depth = linearise_depth(depth, camera.near_plane);
    const float2 uv = (float2(clamped_index.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, depth);
    const float pixel_width_ws = calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, depth);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);
    const float max_visibility_raylen = pixel_width_ws * MAX_VISIBILITY_PIXEL_RANGE;

#if RTGI_USE_QUAD
    // === GS position preload: all 64 threads reconstruct WS positions for the full 12×12 tile ===
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
                    const float pd = push.attach.view_cam_half_res_depth.get()[src];
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

    // === Preload 2x2-blurred luma/rgb into gs_blur_color (geometry-aware) ===
    // One thread per blur-cell (144 entries ≤ 256-thread WG).
    // For each 2×2 block: pick the pixel whose depth is closest to the pseudo-median (robust
    // representative), then only average luma from pixels that lie on the same surface
    // (planar distance < 1 pixel). Store the representative vs position and normal so the
    // per-quad CoV filter can do consistent geometry tests without per-lane divergence.
    {
        const uint flat_i = gtid.y * RTGI_PRE_BLUR_PREPARE_X + gtid.x;
        if (flat_i < uint(GS_LUMA_TOTAL))
        {
            const int bx = int(flat_i) % GS_LUMA_DIM_X;
            const int by = int(flat_i) / GS_LUMA_DIM_X;
            const int2 src_base = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) + int2(bx - QUAD_HALO_CELLS, by - QUAD_HALO_CELLS) * 2;

            // Load all 4 pixels including normals upfront — needed for normal-aware rep selection.
            float  lumas[4];
            float  depths[4];
            float3 ws_pos[4];
            float3 rgbs[4];
            uint   normals_oct[4];
            float3 normals_ws[4];
            float  ray_lengths[4];
            [unroll]
            for (int pi = 0; pi < 4; pi++)
            {
                const int2 src_idx = clamp(src_base + int2(pi & 1, pi >> 1), int2(0, 0), int2(push.size) - 1);
                const int2 gs_pi = int2(bx * 2 + (pi & 1), by * 2 + (pi >> 1));
                const float4 gs_pd = gs_quad_pos_depth[gs_pi.x][gs_pi.y];
                const float pd = gs_pd.w;
                const float  y    = push.attach.diffuse_raw.get()[src_idx].w;
                const float2 cocg = push.attach.diffuse2_raw.get()[src_idx].rg;
                lumas[pi]  = y;
                depths[pi] = pd;
                ws_pos[pi] = gs_pd.xyz;
                rgbs[pi]       = max(float3(y + cocg.x - cocg.y, y + cocg.y, y - cocg.x - cocg.y), 1e-11f);
                normals_oct[pi] = LOAD_NORMAL_OCT(gs_pi);
                normals_ws[pi]  = uncompress_normal_octahedral_32(normals_oct[pi]);
                ray_lengths[pi] = push.attach.ray_length_image.get()[src_idx];
            }

            // Pick representative: normal-majority first, closest depth as tiebreaker.
            // Each non-sky pixel scores +1 for every other non-sky pixel whose normal is within
            // ~45° (dot > 0.707). The pixel with the highest score belongs to the largest
            // surface group; depth breaks ties in favour of the foreground (reversed-Z: larger = closer).
            // Sky pixels (depth==0) get score 0 and are only chosen when all four are sky.
            int normal_score[4] = {0, 0, 0, 0};
            [unroll]
            for (int pi = 0; pi < 4; pi++)
            {
                if (depths[pi] == 0.0f) continue;
                [unroll]
                for (int pj = 0; pj < 4; pj++)
                {
                    if (depths[pj] != 0.0f && dot(normals_ws[pi], normals_ws[pj]) > 0.9f)
                        normal_score[pi]++;
                }
            }

            int rep_i = 0;
            [unroll]
            for (int pi = 1; pi < 4; pi++)
            {
                const bool better_normal = normal_score[pi] > normal_score[rep_i];
                const bool same_normal   = normal_score[pi] == normal_score[rep_i];
                const bool closer        = depths[pi] > depths[rep_i];
                if (better_normal || (same_normal && closer))
                    rep_i = pi;
            }

            const uint rep_normal_oct = normals_oct[rep_i];

            // Accumulate in perceptual (log) space so the blur-cell stores the geometric mean,
            // not the arithmetic mean — avoids Jensen's inequality bias in firefly ceiling and CoV.
            float  luma_acc    = 0.0f;
            float3 rgb_acc     = float3(0.0f, 0.0f, 0.0f);
            float  luma_weight = 0.0f;
            float  rs_acc      = 0.0f;
            float  diffuse_weight = 0.0f;
            if (depths[rep_i] != 0.0f)
            {
                const float3 rep_normal_ws = uncompress_normal_octahedral_32(rep_normal_oct);
                [unroll]
                for (int pi = 0; pi < 4; pi++)
                {
                    const float gw = depths[pi] != 0.0f ? calc_similar_surface_weight(pixel_width_ws_rcp, ws_pos[rep_i], rep_normal_ws, ws_pos[pi], normals_ws[pi]) : 0.0f;
                    luma_acc += linear_to_perceptual(lumas[pi]) * gw;
                    rgb_acc  += linear_to_perceptual(rgbs[pi])  * gw;
                    luma_weight += gw;
                    rs_acc += (1.0f - min(1.0f, ray_lengths[pi] / max(max_visibility_raylen, 1e-8f))) * gw;
                    diffuse_weight += gw;
                }
            }
            const float  rep_log_luma       = linear_to_perceptual(lumas[rep_i]);
            const float3 rep_log_rgb        = linear_to_perceptual(rgbs[rep_i]);
            const float  blur_luma = luma_weight > 0.0f ? luma_acc / luma_weight : rep_log_luma;
            const float3 blur_rgb  = luma_weight > 0.0f ? rgb_acc  / luma_weight : rep_log_rgb;
            gs_blur_color[bx][by]           = float4(blur_rgb, blur_luma);
            gs_quad_position_depth[bx][by]  = float4(ws_pos[rep_i], depths[rep_i]);
            gs_quad_normal_oct[bx][by]      = rep_normal_oct;
            gs_quad_rayshortness[bx][by]    = luma_weight > 0.0f ? rs_acc / luma_weight : 0.0f;
        }
    }

    GroupMemoryBarrierWithGroupSync(); // gs_blur_color, gs_quad_position_depth, gs_quad_normal_oct, gs_quad_rayshortness all visible

    // === Per-pixel quad filter ===
    // Each pixel independently tests the (2*EXTENT+1)^2 surrounding quads against its own
    // position/normal. No cooperative partial-sum stage — one barrier instead of two.
    // Pixels on surface discontinuities within a 2x2 quad now correctly weight neighbours
    // against their own surface rather than the quad representative's surface.
    const int2 pixel_quad_id   = int2(gtid) / 2;
    const int2 pixel_gs_center = pixel_quad_id + int2(QUAD_HALO_CELLS, QUAD_HALO_CELLS);

    float geo_luma_log_acc  = 0.0f;
    float3 geo_rgb_log_acc  = float3(0.0f, 0.0f, 0.0f);
    float geo_weight_acc    = 0.0f;
    float ray_shortness_acc = 0.0f;
    float ff_luma_log_sum   = 0.0f;
    float3 ff_rgb_log_sum   = float3(0.0f, 0.0f, 0.0f);
    int ff_count            = 0;
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
            const float3 sq_normal_ws = uncompress_normal_octahedral_32(gs_quad_normal_oct[sq.x][sq.y]);
            const float  gw = !sq_sky ? calc_similar_surface_weight( pixel_width_ws_rcp, world_position, pixel_face_normal, sq_vs_data.xyz, sq_normal_ws) : 0.0f;

            // gs_blur_color is already in log (perceptual) space.
            const float4 blur_color = gs_blur_color[sq.x][sq.y];
            const float3 log_rgb  = blur_color.xyz;
            const float  log_luma = blur_color.w;
            geo_luma_log_acc  += log_luma * gw;
            geo_rgb_log_acc   += log_rgb  * gw;
            geo_weight_acc    += gw;
            ray_shortness_acc += gs_quad_rayshortness[sq.x][sq.y] * gw;

            if (!sq_sky)
            {
                ff_luma_log_sum += log_luma;
                ff_rgb_log_sum  += log_rgb;
                ff_count++;
            }
        }
    }

    // Center quad: 4 individual pixel taps at full (non-quad) resolution.
    // The quad representative may disagree with edge pixels inside the quad — testing the
    // 4 raw pixels eliminates false inf sentinels at depth discontinuities.
    // Apply linear_to_perceptual here (raw pixels, unlike gs_blur_color which pre-logs).
    float center_weight_acc = 0.0f;
    float4 center_quad_diffuse_acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 center_quad_diffuse2_acc = float2(0.0f, 0.0f);
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
            const float  tap_gw      = calc_similar_surface_weight(pixel_width_ws_rcp, world_position, pixel_face_normal, tap_pos_ws, tap_nrm_ws);
            const float4 tap_diffuse  = push.attach.diffuse_raw.get()[tap_idx];
            const float2 tap_diffuse2 = push.attach.diffuse2_raw.get()[tap_idx].rg;
            const float  tap_y       = max(tap_diffuse.w, 1e-3f);
            const float2 tap_cocg    = tap_diffuse2;
            const float3 tap_rgb     = max(float3(tap_y + tap_cocg.x - tap_cocg.y, tap_y + tap_cocg.y, tap_y - tap_cocg.x - tap_cocg.y), 1e-3f);
            const float  tap_log_y   = linear_to_perceptual(tap_y);
            const float3 tap_log_rgb = linear_to_perceptual(tap_rgb);
            const float  tap_close   = 1.0f - min(1.0f, push.attach.ray_length_image.get()[tap_idx] / max(max_visibility_raylen, 1e-8f));

            geo_luma_log_acc  += tap_log_y   * tap_gw;
            geo_rgb_log_acc   += tap_log_rgb  * tap_gw;
            geo_weight_acc    += tap_gw;
            ray_shortness_acc += tap_close    * tap_gw;
            ff_luma_log_sum   += tap_log_y;
            ff_rgb_log_sum    += tap_log_rgb;
            ff_count++;
            center_weight_acc += tap_gw;
            center_quad_diffuse_acc += tap_diffuse * tap_gw;
            center_quad_diffuse2_acc += tap_diffuse2 * tap_gw;
        }
    }
    const float  center_blend           = center_weight_acc * 0.25f; // 0-1: fraction of center pixels on same surface
    const bool   pixel_matches_quad     = center_weight_acc > 0.0f;
    const float4 center_quad_diffuse_mean  = center_weight_acc > 0.0f ? center_quad_diffuse_acc / center_weight_acc : push.attach.diffuse_raw.get()[clamped_index];
    const float2 center_quad_diffuse2_mean = center_weight_acc > 0.0f ? center_quad_diffuse2_acc / center_weight_acc : push.attach.diffuse2_raw.get()[clamped_index].rg;

    const float  quad_luma_mean_geometric = geo_weight_acc > 0.0f ? perceptual_to_linear(geo_luma_log_acc / geo_weight_acc) : 0.0f;
    const float3 quad_rgb_mean_geometric  = geo_weight_acc > 0.0f ? perceptual_to_linear(geo_rgb_log_acc  / geo_weight_acc) : float3(0.0f, 0.0f, 0.0f);
    const float  firefly_luma_mean = ff_count > 0 ? perceptual_to_linear(ff_luma_log_sum / float(ff_count)) : quad_luma_mean_geometric;
    const float3 firefly_rgb_mean  = ff_count > 0 ? perceptual_to_linear(ff_rgb_log_sum  / float(ff_count)) : quad_rgb_mean_geometric;
    const float  total_quad_taps = float(QUAD_FILTER_DIM * QUAD_FILTER_DIM - 1 + 4);  // surrounding quads + 4 center pixels
    const float  footprint_quality_quad = min(geo_weight_acc / total_quad_taps, 1.0f);
    const float  ray_shortness_mean_quad = geo_weight_acc > 0.0f ? ray_shortness_acc / geo_weight_acc : 0.0f;
#else
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_normals.get()[clamped_index]);
    const float4 _nq_wp = mul(camera->inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = _nq_wp.xyz / _nq_wp.w;
    // Precalculate surrounding view space position data for geometric testing
    {
        [unroll]
        for (uint group_iter_x = 0; group_iter_x < round_up_div(PRELOAD_SIZE_X, RTGI_PRE_BLUR_PREPARE_X); ++group_iter_x)
        {
            [unroll]
            for (uint group_iter_y = 0; group_iter_y < round_up_div(PRELOAD_SIZE_Y, RTGI_PRE_BLUR_PREPARE_Y); ++group_iter_y)
            {
                const int2 in_preload_index = int2(group_iter_x * RTGI_PRE_BLUR_PREPARE_X, group_iter_y * RTGI_PRE_BLUR_PREPARE_Y) + int2(gtid);
                const int2 src_group_base_index = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) - TOTAL_FILTER_REACH;
                const int2 src_index = src_group_base_index + in_preload_index;
                int2 clamped_src_index = clamp(src_index, int2(0,0), int2(push.size - 1));
                const int2 max_index = push.size - 1;
                clamped_src_index = flip_oob_index(clamped_src_index, max_index);

                if (all(in_preload_index < int2(PRELOAD_SIZE_X, PRELOAD_SIZE_Y)))
                {
                    const float preload_depth = push.attach.view_cam_half_res_depth.get()[clamped_src_index];
                    const float2 preload_uv = (float2(clamped_src_index) + 0.5f) * inv_half_res_render_target_size;
                    const float4 preload_pos_pre_div = mul(camera.inv_view_proj, float4(preload_uv * 2.0f - 1.0f, preload_depth, 1.0f));
                    const float3 preload_ws_pos = preload_pos_pre_div.xyz / preload_pos_pre_div.w;
                    gs_sample_positions[in_preload_index.x][in_preload_index.y] = float4(preload_ws_pos, preload_depth);

                    const float3 alb = push.attach.view_cam_half_res_albedo.get()[clamped_src_index].rgb;
                    const uint3 alb_q = (uint3)(saturate(alb) * 255.0f + 0.5f);
                    gs_albedo_rgb[in_preload_index.x][in_preload_index.y] = alb_q.r | (alb_q.g << 8) | (alb_q.b << 16);
                    gs_sample_normal_ws_nonquad[in_preload_index.x][in_preload_index.y] = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_normals.get()[clamped_src_index]);
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); // gs_sample_positions + gs_albedo_rgb all visible
    
    // Geometric mean is used to suppress fireflies
    float luma_mean_geometric_acc = 0.0f;
    float3 rgb_mean_geometric_acc = float3(0.0f, 0.0f, 0.0f);
    float valid_geometric_mean_samples = 0.0f;
    // Per-channel log-space sq accumulators for signal variance — same tap set as geometric mean.
    // Max across channels prevents single-channel noise from being masked by smooth luma.
    float3 rgb_log_sq_acc = float3(0.0f, 0.0f, 0.0f);

    // Variance, ray length and valid footprint samples are used for pre blur filter guiding later
    float y_mean_acc = 0.0f;
    float y_variance_acc = 0.0f;
    float y_max = 0.0f;
    float ray_shortness_acc = 0.0f;
    float ray_shortness_weight_acc = 0.0f;
    float valid_footprint_samples = 0.0f;
    float geo_weighted_luma_log_acc = 0.0f;
    float geo_weighted_luma_weight_acc = 0.0f;

    float3 albedo_mean_acc = float3(0.0f, 0.0f, 0.0f);

    // Per-pixel albedo gradient estimation: accumulate |neighbor_alb - center_alb| from groupshared
    const float3 pp_center_alb = unpack_rgb(gs_albedo_rgb[grad_idx.x][grad_idx.y]);
    float pp_alb_diff_acc = 0.0f;
    float pp_alb_diff_count = 0.0f;
    // Coefficient of variation accumulators (same 3x3 window)
    float pp_alb_lum_acc = 0.0f;
    float pp_alb_lum_sq_acc = 0.0f;

    // The raw signal is pre blurred in a star shape with 5 taps to conserve energy from the firefly filter.
    float4 center_blurred_diffuse_acc = float4(0,0,0,0);
    float2 center_blurred_diffuse2_acc = float2(0,0);
    float center_blurred_weight_acc = 0.0f;

    [unroll]
    for (int x = -TOTAL_FILTER_REACH; x <= TOTAL_FILTER_REACH; ++x)
    {
        [unroll]
        for (int y = -TOTAL_FILTER_REACH; y <= TOTAL_FILTER_REACH; ++y)
        {
            const int2 max_index = push.size - 1;
            int2 load_idx = int2(x,y) * FILTER_STRIDE + int2(clamped_index);
            load_idx = flip_oob_index(load_idx, max_index);

            const int2 preload_index = src_to_preload_index(load_idx, gid);
            float4 sample_diffuse = push.attach.diffuse_raw.get()[load_idx];                // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            float2 sample_diffuse2 = push.attach.diffuse2_raw.get()[load_idx];              // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            const float sample_ray_length = push.attach.ray_length_image.get()[load_idx];   // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            const float4 preload_v = gs_sample_positions[preload_index.x][preload_index.y];
            const float3 sample_ws_pos = preload_v.xyz;
            const float sample_depth = preload_v.w;
            float sample_y = sample_diffuse.w;
            sample_y = max(sample_y, 1e-3f);          // Values below this start to break float16 precision, have to clamp radiance up to that value as a minimum for statistic analysis.
            const bool is_sky = sample_depth == 0.0f;
            if (is_sky)
            {
                continue;
            }

            // Geometry weight for this tap via surface similarity in world space (no matmul needed)
            const float3 sample_normal_ws_here = gs_sample_normal_ws_nonquad[preload_index.x][preload_index.y];
            const float geo_weight = calc_similar_surface_weight(pixel_width_ws_rcp, world_position, pixel_face_normal, sample_ws_pos, sample_normal_ws_here);

            // Geometry-weighted ray shortness over full 5x5 neighborhood
            ray_shortness_acc += (1.0f - min(1.0f, sample_ray_length * rcp(max_visibility_raylen))) * geo_weight;
            ray_shortness_weight_acc += geo_weight;


            float geometric_mean_acc_value = linear_to_perceptual(sample_y);

            // Geometry-weighted geometric luma mean for geo_mean_perceptual_image
            geo_weighted_luma_log_acc += geometric_mean_acc_value * geo_weight;
            geo_weighted_luma_weight_acc += geo_weight;

            if (!is_part_of_center_blur(int2(x,y)))
            {
                luma_mean_geometric_acc += geometric_mean_acc_value;
                // RGB parallel path: reconstruct R,G,B from Y/Co/Cg (standard YCoCg inverse)
                const float sample_co = sample_diffuse2.x;
                const float sample_cg = sample_diffuse2.y;
                const float3 sample_rgb = max(float3(
                    sample_y + sample_co - sample_cg,
                    sample_y + sample_cg,
                    sample_y - sample_co - sample_cg
                ), 1e-3f);
                const float3 rgb_log_value = linear_to_perceptual(sample_rgb);
                rgb_mean_geometric_acc += rgb_log_value;
                rgb_log_sq_acc += rgb_log_value * rgb_log_value;
                valid_geometric_mean_samples += 1.0f;
            }

            if (all(abs(int2(x,y)) <= TOTAL_INNER_FILTER_REACH))
            {
                const float GEO_WEIGHT_THRESHOLD = 1.0f * FILTER_STRIDE;
                const float plane_distance = calc_plane_distance(world_position, pixel_face_normal, sample_ws_pos) * pixel_width_ws_rcp;
                const float geometric_weight = geo_weight;//abs(plane_distance) < 1.0f ? 1.0f : 0.0f;

                if (is_part_of_center_blur(int2(x,y)))
                {
                    center_blurred_weight_acc += geometric_weight;
                    center_blurred_diffuse_acc += geometric_weight * sample_diffuse;
                    center_blurred_diffuse2_acc += geometric_weight * sample_diffuse2;
                }

                y_mean_acc += sample_y * geometric_weight;
                y_variance_acc += square(sample_y) * geometric_weight;
                y_max = max(y_max, sample_y) * geometric_weight;
                valid_footprint_samples += geometric_weight;

                const float3 sample_albedo = push.attach.view_cam_half_res_albedo.get()[load_idx].rgb;
                albedo_mean_acc += sample_albedo * geometric_weight;


                if (geometric_weight > 0.0f)
                {
                    const float3 pp_neighbor_alb = unpack_rgb(gs_albedo_rgb[preload_index.x][preload_index.y]);
                    pp_alb_diff_acc += dot(abs(pp_neighbor_alb - pp_center_alb), (1.0f / 3.0f).xxx);
                    pp_alb_diff_count += 1.0f;
                    const float pp_neighbor_lum = dot(pp_neighbor_alb, float3(0.2126f, 0.7152f, 0.1722f));
                    pp_alb_lum_acc += pp_neighbor_lum;
                    pp_alb_lum_sq_acc += square(pp_neighbor_lum);
                }
            }
        }
    }
#endif

    let rtgi = push.attach.globals->rtgi_settings;

    // --- Footprint Quality ---
    float footprint_quality = 1.0f;
    {
#if RTGI_USE_QUAD
        footprint_quality = footprint_quality_quad;
#else
        footprint_quality = valid_footprint_samples * rcp(float(INNER_FILTER_TAPS_TOTAL));
#endif
    }

#if !RTGI_USE_QUAD
    // --- Geometric means (used by both guide output switch and firefly filter) ---
    const float luma_mean_geometric = perceptual_to_linear(luma_mean_geometric_acc * rcp(valid_geometric_mean_samples));
    const float3 rgb_mean_geometric = perceptual_to_linear(rgb_mean_geometric_acc * rcp(valid_geometric_mean_samples));

    // --- Surface Detail ---
    const float pp_lum_mean = pp_alb_lum_acc / (pp_alb_diff_count + 1e-8f);
    const float pp_lum_sq_mean = pp_alb_lum_sq_acc / (pp_alb_diff_count + 1e-8f);
    const float pp_lum_variance = max(0.0f, pp_lum_sq_mean - square(pp_lum_mean));
    const float pp_lim_std_dev = saturate(sqrt(pp_lum_variance) - pp_lum_mean * 0.01f);
    const float surface_albedo_cv = saturate(pp_lim_std_dev / max(pp_lum_mean, 0.05f));
    const float3 rgb_log_mean = rgb_mean_geometric_acc * rcp(valid_geometric_mean_samples + 1e-8f);
    const float3 rgb_log_variance = max(float3(0,0,0), rgb_log_sq_acc * rcp(valid_geometric_mean_samples + 1e-8f) - rgb_log_mean * rgb_log_mean);
    const float3 rgb_log_std_dev = sqrt(rgb_log_variance);
    const float y_log_std_dev = max(max(rgb_log_std_dev.r, rgb_log_std_dev.g), rgb_log_std_dev.b);
#endif // !RTGI_USE_QUAD

    // --- Guide outputs: switch between per-pixel loop and gs quad ---
#if RTGI_USE_QUAD
    const float  out_mean      = quad_luma_mean_geometric;
    const float3 out_rgb_mean  = quad_rgb_mean_geometric;
#else
    const float  out_mean      = luma_mean_geometric;
    const float3 out_rgb_mean  = rgb_mean_geometric;
#endif

    // --- Firefly Filter ---
    float4 filtered_diffuse;
    float2 filtered_diffuse2;
    float firefly_energy_factor = 1.0f;
#if !RTGI_USE_QUAD
    float y_std_dev;
    float y_luma_mean;
    float luma_mean_geometric_over_linear = 1.0f;
#endif
    {
        const float EPSILON = 1e-8f;
#if RTGI_USE_QUAD
        const float4 raw_diffuse  = push.attach.diffuse_raw.get()[clamped_index];
        const float2 raw_diffuse2 = push.attach.diffuse2_raw.get()[clamped_index].rg;
        const float4 center_blurred_diffuse  = lerp(raw_diffuse,  center_quad_diffuse_mean,  center_blend);
        const float2 center_blurred_diffuse2 = lerp(raw_diffuse2, center_quad_diffuse2_mean, center_blend);
#else
        const float4 center_blurred_diffuse = center_blurred_diffuse_acc * rcp(center_blurred_weight_acc + EPSILON);
        const float2 center_blurred_diffuse2 = center_blurred_diffuse2_acc * rcp(center_blurred_weight_acc + EPSILON);
#endif

        const bool use_center_blur = rtgi.firefly_filter_enabled != 0 && rtgi.firefly_center_blur_enabled != 0;
        if (use_center_blur)
        {
            filtered_diffuse = center_blurred_diffuse;
            filtered_diffuse2 = center_blurred_diffuse2;
        }
        else
        {
            filtered_diffuse = push.attach.diffuse_raw.get()[clamped_index];
            filtered_diffuse2 = push.attach.diffuse2_raw.get()[clamped_index];
        }

#if !RTGI_USE_QUAD
        y_luma_mean = y_mean_acc * rcp(valid_footprint_samples);
        const float y_variance = (y_variance_acc * rcp(valid_footprint_samples)) - square(y_luma_mean);
        y_std_dev = sqrt(y_variance);
#endif

#if RTGI_USE_QUAD
        const float ceiling_factor = max(1.0f, rtgi.firefly_filter_ceiling * footprint_quality * (pixel_matches_quad ? 1.0f : 0.1f));
#else
        const float ceiling_factor = max(1.0f, rtgi.firefly_filter_ceiling * footprint_quality);
#endif

#if !RTGI_USE_QUAD
        luma_mean_geometric_over_linear = out_mean / y_luma_mean;
#endif

        // RGB parallel clamp: clamp R,G,B channels using geometric means, convert back to YCoCg clamp factors
        const float filt_co = filtered_diffuse2.x;
        const float filt_cg = filtered_diffuse2.y;
        const float filt_y  = filtered_diffuse.w;
        const float3 filtered_rgb = float3(
            filt_y + filt_co - filt_cg,
            filt_y + filt_cg,
            filt_y - filt_co - filt_cg
        );

#if RTGI_USE_QUAD
        // Use unweighted neighbourhood means so isolated detail pixels have a full reference.
        const float  geo_mean_for_ceiling = firefly_luma_mean;
        const float3 rgb_mean_for_ceiling = firefly_rgb_mean;
#else
        const float  geo_mean_for_ceiling     = out_mean;
        const float3 rgb_mean_for_ceiling     = out_rgb_mean;
#endif
        const float geometric_y_clamp_factor = min((geo_mean_for_ceiling * ceiling_factor) / (EPSILON + filtered_diffuse.w), 1.0f);
        // Clamp each RGB channel, then convert clamped RGB back to YCoCg to get per-channel clamp factors
        const float3 clamped_rgb = min(filtered_rgb, rgb_mean_for_ceiling * ceiling_factor);
        const float y_from_rgb   = (clamped_rgb.r + 2.0f * clamped_rgb.g + clamped_rgb.b) * 0.25f;
        const float co_from_rgb  = (clamped_rgb.r - clamped_rgb.b) * 0.5f;
        const float cg_from_rgb  = (-clamped_rgb.r + 2.0f * clamped_rgb.g - clamped_rgb.b) * 0.25f;
        const float y_rgb_clamp_factor  = min(y_from_rgb  / (filt_y  + EPSILON), 1.0f);
        const float co_rgb_clamp_factor = abs(filt_co) > EPSILON ? clamp(co_from_rgb / filt_co, 0.0f, 1.0f) : 1.0f;
        const float cg_rgb_clamp_factor = abs(filt_cg) > EPSILON ? clamp(cg_from_rgb / filt_cg, 0.0f, 1.0f) : 1.0f;

        // Min-RGB: hue-preserving scale — ceiling scaled inversely by channel brightness relative to max channel mean
        // (a channel 10x darker than the brightest channel gets 10x the ceiling, cancels out to max_channel_mean * ceiling_factor for all channels)
        const float max_channel_mean = max(max(rgb_mean_for_ceiling.r, rgb_mean_for_ceiling.g), rgb_mean_for_ceiling.b);
        const float3 excess = filtered_rgb / max(max_channel_mean * ceiling_factor, EPSILON);
        const float max_excess = max(max(excess.r, excess.g), excess.b);
        const float min_rgb_scale = 1.0f / max(max_excess, 1.0f);

        if (rtgi.firefly_filter_enabled != 0)
        {
            float final_y_clamp;
            float final_co_clamp;
            float final_cg_clamp;
            if (rtgi.firefly_clamp_mode == 0) // multichromatic: hue-preserving single scalar
            {
                final_y_clamp  = min_rgb_scale;
                final_co_clamp = min_rgb_scale;
                final_cg_clamp = min_rgb_scale;
            }
            else // monochromatic (mode 1): luma clamp
            {
                final_y_clamp  = min(geometric_y_clamp_factor, y_rgb_clamp_factor);
                final_co_clamp = co_rgb_clamp_factor;
                final_cg_clamp = cg_rgb_clamp_factor;
            }
            filtered_diffuse    *= final_y_clamp;
            filtered_diffuse2.x *= final_co_clamp;
            filtered_diffuse2.y *= final_cg_clamp;
            if (rtgi.firefly_energy_compensation_enabled != 0)
            {
                firefly_energy_factor = 1.0f / max(1e-6f, final_y_clamp);
            }
        }
    }

    // --- Raylength Guide ---
#if RTGI_USE_QUAD
    const float ray_shortness_mean = ray_shortness_mean_quad;
#else
    const float ray_shortness_mean = ray_shortness_acc * rcp(ray_shortness_weight_acc + 1e-8f);
#endif
    const float raylength_filter_guide = (1.0f - sqrt(sqrt(ray_shortness_mean))) * footprint_quality;

    // Pixels that don't belong to their quad's representative surface get inf mean
    // so the pre-blur can skip variance weighting for them.
#if RTGI_USE_QUAD
    float written_mean = linear_to_perceptual(out_mean);
    //if (!pixel_matches_quad)
        //written_mean = 1.0f / 0.0f;
#else
    const float geo_weighted_luma_mean = geo_weighted_luma_weight_acc > 1e-8f
        ? perceptual_to_linear(geo_weighted_luma_log_acc / geo_weighted_luma_weight_acc)
        : out_mean;
    float written_mean = linear_to_perceptual(geo_weighted_luma_mean);
#endif


    // debug_image_tile_draw(dbg, 0,  dtid, float4(TurboColormap(written_mean+2), 2.0f), 2);
    // debug_image_tile_draw(dbg, 0,  dtid, float4(TurboColormap(footprint_quality), 2.0f), 2);
    // debug_image_tile_draw(dbg, -1, dtid, lerp(float4(1,0,0,2), float4(0,1,0,2), quad_blend), 2);
    //if (written_mean == -1.0f) debug_image_tile_draw(dbg, -1, dtid, float4(1, 0, 0, 2), 2);

    push.attach.pre_filtered_diffuse_image.get()[dtid] = filtered_diffuse;
    push.attach.pre_filtered_diffuse2_image.get()[dtid] = filtered_diffuse2;
    push.attach.firefly_factor_image.get()[dtid] = firefly_energy_factor;
    push.attach.geo_mean_perceptual_image.get()[dtid] = written_mean;
    push.attach.filter_guide_image.get()[dtid] = raylength_filter_guide;
}