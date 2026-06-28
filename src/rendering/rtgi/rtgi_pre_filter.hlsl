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
static const float MAX_VISIBILITY_PIXEL_RANGE  = 48.0f; // ~1.5x the max denoiser width

func linear_to_perceptual(float v) -> float
{
    return log(max(v, 1e-8f) * PERCEPTUAL_SPACE_MULTIPLIER);
}
__generic<uint N>
func linear_to_perceptual(vector<float, N> v) -> vector<float, N>
{
    return log(max(v, 1e-8f) * PERCEPTUAL_SPACE_MULTIPLIER);
}

func perceptual_to_linear(float v) -> float
{
    return exp(v) / PERCEPTUAL_SPACE_MULTIPLIER;
}
__generic<uint N>
func perceptual_to_linear(vector<float, N> v) -> vector<float, N>
{
    return (exp(v)) / PERCEPTUAL_SPACE_MULTIPLIER;
}

// The center blur is used to preserve firefly energy by flat filtering all pixels in a star kernel covering 5 pixels.
func is_part_of_center_blur(int2 index) -> bool
{
    return (abs(index.x) + abs(index.y) <= 1);
}

static const int FILTER_STRIDE = 1;
static const int TOTAL_FILTER_REACH = 2;
static const int TOTAL_INNER_FILTER_REACH = 1;                                         // Inner filter calculates geometry aware metrics: footprint valid samples ratio, star blur, ray length average
static const int FILTER_TAPS_TOTAL = (TOTAL_FILTER_REACH * 2 + 1) * (TOTAL_FILTER_REACH * 2 + 1);
static const int INNER_FILTER_TAPS_TOTAL = (TOTAL_INNER_FILTER_REACH * 2 + 1) * (TOTAL_INNER_FILTER_REACH * 2 + 1);
static const int STAR_BLUR_TAPS = 5;                                                    // 5 generally yields the best cost to use ratio.
static const int GEOMETRIC_MEAN_TAPS_TOTAL = FILTER_TAPS_TOTAL - STAR_BLUR_TAPS;              // 20 geometric mean taps are a good ratio against the 5 center blur taps.

static const int PRELOAD_SIZE_X = RTGI_PRE_BLUR_PREPARE_X + TOTAL_FILTER_REACH * 2;
static const int PRELOAD_SIZE_Y = RTGI_PRE_BLUR_PREPARE_X + TOTAL_FILTER_REACH * 2;
groupshared float4 gs_sample_value_vs[PRELOAD_SIZE_X][PRELOAD_SIZE_Y]; // .xyz = sample vs space pos, .w = depth
// Full PRELOAD_SIZE (20x20) gives a 2-pixel border, supporting stride 1 and stride 2 gradients
// RGB packed as R8G8B8 in the low 24 bits of a uint
groupshared uint gs_albedo_rgb[PRELOAD_SIZE_X][PRELOAD_SIZE_Y];

// --- 2x2-blurred luma grid and per-quad CoV ---
// gs_luma_blurred covers (WG/2 + 4) x (WG/2 + 4) = 12x12 entries.
// Each entry is the average Y (luma) of a 2x2 block of diffuse_raw, stride 2.
// Entries 0–1 and 10–11 extend 2 blur-cells (= 4 input pixels) past each workgroup edge.
static const int GS_LUMA_DIM_X = RTGI_PRE_BLUR_PREPARE_X / 2 + 4;   // 12
static const int GS_LUMA_DIM_Y = RTGI_PRE_BLUR_PREPARE_Y / 2 + 4;   // 12
static const int GS_LUMA_TOTAL = GS_LUMA_DIM_X * GS_LUMA_DIM_Y;     // 144
static const int NUM_QUADS_X   = RTGI_PRE_BLUR_PREPARE_X / 2;        // 8
static const int NUM_QUADS_Y   = RTGI_PRE_BLUR_PREPARE_Y / 2;        // 8
static const int NUM_QUADS     = NUM_QUADS_X * NUM_QUADS_Y;           // 64

groupshared float  gs_luma_blurred[GS_LUMA_DIM_X][GS_LUMA_DIM_Y];
groupshared float3 gs_rgb_blurred[GS_LUMA_DIM_X][GS_LUMA_DIM_Y];    // geometry-aware averaged linear RGB per blur-cell
groupshared float4 gs_quad_vs[GS_LUMA_DIM_X][GS_LUMA_DIM_Y];        // .xyz = representative vs pos, .w = depth (sky→0)
groupshared uint   gs_quad_normal_oct[GS_LUMA_DIM_X][GS_LUMA_DIM_Y]; // packed octahedral face normal per blur-cell
groupshared float  gs_quad_accept_count[GS_LUMA_DIM_X][GS_LUMA_DIM_Y]; // how many of the 4 blur-cell pixels passed the geometry test
groupshared float2 gs_luma_partial[NUM_QUADS][4];  // .x = partial log-luma sum, .y = valid_count  (4 lanes per quad)
groupshared float3 gs_rgb_partial[NUM_QUADS][4];   // .xyz = log(R), log(G), log(B) partial sums (4 lanes per quad)
groupshared float  gs_quad_luma_mean[NUM_QUADS];   // geometric luma mean per quad
groupshared float3 gs_rgb_mean_result[NUM_QUADS];  // .xyz = linear RGB geometric mean
#if RTGI_USE_QUAD
groupshared float  gs_quad_rayshortness[GS_LUMA_DIM_X][GS_LUMA_DIM_Y]; // mean ray shortness [0,1] per blur-cell
groupshared float  gs_rayshortness_partial[NUM_QUADS][4];               // partial sums, one per CoV lane
groupshared float  gs_quad_rayshortness_result[NUM_QUADS];              // sqrt(sqrt(mean)) result per quad
#endif
#if RTGI_USE_QUAD
groupshared float4 gs_quad_diffuse[GS_LUMA_DIM_X][GS_LUMA_DIM_Y];     // geometry-weighted mean SH diffuse per blur-cell
groupshared float2 gs_quad_diffuse2[GS_LUMA_DIM_X][GS_LUMA_DIM_Y];    // geometry-weighted mean Co/Cg per blur-cell
#endif
#if RTGI_USE_QUAD
groupshared float  gs_footprint_partial[NUM_QUADS][4];                  // partial accept_count/4 sums, one per CoV lane
groupshared float  gs_quad_footprint_result[NUM_QUADS];                 // mean footprint quality over 5x5 neighbourhood
#endif

// 24 offsets of a 5x5 neighbourhood (center excluded), row-major.
// Partitioned 6-per-lane: lane 0 → [0..5], lane 1 → [6..11], lane 2 → [12..17], lane 3 → [18..23].
static const int2 COV_SURROUND_OFFS[24] = {
    int2(-2,-2), int2(-1,-2), int2( 0,-2), int2( 1,-2), int2( 2,-2),  // dy=-2 (5)
    int2(-2,-1),                                                        // dy=-1 start (lane 0 end)
    int2(-1,-1), int2( 0,-1), int2( 1,-1), int2( 2,-1),               // dy=-1 cont
    int2(-2, 0), int2(-1, 0),                                          // dy= 0, skip center (lane 1 end)
    int2( 1, 0), int2( 2, 0),                                          // dy= 0 cont
    int2(-2, 1), int2(-1, 1), int2( 0, 1), int2( 1, 1),               // dy=+1 (lane 2 end)
    int2( 2, 1),                                                        // dy=+1 end
    int2(-2, 2), int2(-1, 2), int2( 0, 2), int2( 1, 2), int2( 2, 2), // dy=+2 (5)
};

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

    // === Preload 2x2-blurred luma into gs_luma_blurred (geometry-aware) ===
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
            const int2 src_base = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) + int2(bx - 2, by - 2) * 2;

            // Load all 4 pixels including normals upfront — needed for normal-aware rep selection.
            float  lumas[4];
            float  depths[4];
            float3 vs_pos[4];
            float3 rgbs[4];
            uint   normals_oct[4];
            float3 normals_ws[4];
            float  ray_lengths[4];
            float4 diffuses[4];
            float2 diffuse2s[4];
            [unroll]
            for (int pi = 0; pi < 4; pi++)
            {
                const int2 src_idx = clamp(src_base + int2(pi & 1, pi >> 1), int2(0, 0), int2(push.size) - 1);
                const float pd = push.attach.view_cam_half_res_depth.get()[src_idx];
                const float2 puv = (float2(src_idx) + 0.5f) * inv_half_res_render_target_size;
                const float4 pvs_pre_div = mul(camera.inv_proj, float4(puv * 2.0f - 1.0f, pd, 1.0f));
                const float  y    = push.attach.diffuse_raw.get()[src_idx].w;
                const float2 cocg = push.attach.diffuse2_raw.get()[src_idx].rg;
                lumas[pi]      = y;
                depths[pi]     = pd;
                vs_pos[pi]     = pvs_pre_div.xyz * rcp(pvs_pre_div.w);
                rgbs[pi]       = max(float3(y + cocg.x - cocg.y, y + cocg.y, y - cocg.x - cocg.y), 1e-3f);
                normals_oct[pi] = push.attach.view_cam_half_res_normals.get()[src_idx];
                normals_ws[pi]  = uncompress_normal_octahedral_32(normals_oct[pi]);
                ray_lengths[pi] = push.attach.ray_length_image.get()[src_idx];
                diffuses[pi]  = push.attach.diffuse_raw.get()[src_idx];
                diffuse2s[pi] = push.attach.diffuse2_raw.get()[src_idx].rg;
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
            float4 diffuse_acc  = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float2 diffuse2_acc = float2(0.0f, 0.0f);
            float  diffuse_weight = 0.0f;
            if (depths[rep_i] != 0.0f)
            {
                const float3 rep_normal_ws = uncompress_normal_octahedral_32(rep_normal_oct);
                const float3 rep_normal_vs = mul(camera.view, float4(rep_normal_ws, 0.0f)).xyz;
                const float  rep_px_ws_size_rcp = rcp(ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, depths[rep_i]));
                const float  rep_max_raylen = MAX_VISIBILITY_PIXEL_RANGE / rep_px_ws_size_rcp;
                [unroll]
                for (int pi = 0; pi < 4; pi++)
                {
                    const float3 sample_normal_vs = mul(camera.view, float4(normals_ws[pi], 0.0f)).xyz;
                    const float gw = depths[pi] != 0.0f ? surface_weight_dist_limited(inv_half_res_render_target_size, camera.near_plane, depths[rep_i], vs_pos[rep_i], rep_normal_vs, vs_pos[pi], sample_normal_vs) : 0.0f;
                    luma_acc += linear_to_perceptual(lumas[pi]) * gw;
                    rgb_acc  += linear_to_perceptual(rgbs[pi])  * gw;
                    luma_weight += gw;
                    rs_acc += (1.0f - min(1.0f, ray_lengths[pi] / max(rep_max_raylen, 1e-8f))) * gw;
                    diffuse_acc    += diffuses[pi]  * gw;
                    diffuse2_acc   += diffuse2s[pi] * gw;
                    diffuse_weight += gw;
                }
            }
            const float  rep_log_luma = linear_to_perceptual(lumas[rep_i]);
            const float3 rep_log_rgb  = linear_to_perceptual(rgbs[rep_i]);
            gs_luma_blurred[bx][by]    = luma_weight > 0.0f ? luma_acc / luma_weight : rep_log_luma;
            gs_rgb_blurred[bx][by]     = luma_weight > 0.0f ? rgb_acc  / luma_weight : rep_log_rgb;
            gs_quad_vs[bx][by]         = float4(vs_pos[rep_i], depths[rep_i]);
            gs_quad_normal_oct[bx][by] = rep_normal_oct;
            gs_quad_accept_count[bx][by] = luma_weight;
#if RTGI_USE_QUAD
            gs_quad_rayshortness[bx][by] = luma_weight > 0.0f ? rs_acc / luma_weight : 0.0f;
            gs_quad_diffuse[bx][by]  = diffuse_weight > 0.0f ? diffuse_acc  / diffuse_weight : diffuses[rep_i];
            gs_quad_diffuse2[bx][by] = diffuse_weight > 0.0f ? diffuse2_acc / diffuse_weight : diffuse2s[rep_i];
#endif
        }
    }

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
                    const float2 uv = (float2(clamped_src_index.xy) + 0.5f) * inv_half_res_render_target_size;
                    const float3 sample_value_ndc = float3(uv * 2.0f - 1.0f, preload_depth);
                    const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
                    const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);
                    gs_sample_value_vs[in_preload_index.x][in_preload_index.y] = float4(sample_value_vs, preload_depth);

                    const float3 alb = push.attach.view_cam_half_res_albedo.get()[clamped_src_index].rgb;
                    const uint3 alb_q = (uint3)(saturate(alb) * 255.0f + 0.5f);
                    gs_albedo_rgb[in_preload_index.x][in_preload_index.y] = alb_q.r | (alb_q.g << 8) | (alb_q.b << 16);
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); // gs_luma_blurred + gs_sample_value_vs + gs_albedo_rgb all visible

    // === Per-quad CoV over 5x5 blurred-luma neighbourhood ===
    // Each 2x2 thread block (quad) cooperates: the 4 lanes split the 24 surrounding entries 6-each,
    // then lane 0 of the quad combines all partials + the center to produce a single CoV.
    // All 4 threads in the quad draw the same result to a debug tile.
    {
        const int2 quad_id   = int2(gtid) / 2;
        const int  quad_lane = int(gtid.y & 1u) * 2 + int(gtid.x & 1u);   // 0..3
        const int  flat_quad = quad_id.y * NUM_QUADS_X + quad_id.x;
        const int2 gs_center = quad_id + int2(2, 2);   // +2 border offset into gs_luma_blurred

        // Read per-quad representative geometry from the preloaded gs_quad_vs/gs_quad_normal_oct.
        // All 4 lanes in the quad share the same gs_center → consistent center geometry.
        const float4 center_quad_vs_data = gs_quad_vs[gs_center.x][gs_center.y];
        const float3 center_vs           = center_quad_vs_data.xyz;
        const float  center_depth        = center_quad_vs_data.w;
        const float3 center_face_normal_ws = uncompress_normal_octahedral_32(gs_quad_normal_oct[gs_center.x][gs_center.y]);
        const float3 center_vs_normal    = mul(camera.view, float4(center_face_normal_ws, 0.0f)).xyz;
        const float  center_px_ws_size_rcp = rcp(ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, center_depth));

        // Each lane accumulates its 6 surrounding blur-cells in log space.
        // Geometry test uses gs_quad_vs which covers the full 12×12 grid — no clamping needed.
        // center_depth check hoisted outside — sky quads skip the loop entirely.
        const int lane_start = quad_lane * 6;
        float partial_sum    = 0.0f;
        float partial_count  = 0.0f;
        float3 rgb_partial_sum = float3(0.0f, 0.0f, 0.0f);
#if RTGI_USE_QUAD
        float partial_rayshortness = 0.0f;
#endif
#if RTGI_USE_QUAD
        float partial_footprint = 0.0f;
#endif
        if (center_depth != 0.0f)
        {
            [unroll]
            for (int k = 0; k < 6; k++)
            {
                const int2 blur_off      = COV_SURROUND_OFFS[lane_start + k];
                const int2 sample_gs_idx = gs_center + blur_off;
                const float4 sample_vs_data = gs_quad_vs[sample_gs_idx.x][sample_gs_idx.y];
                const float  sample_depth   = sample_vs_data.w;
                const float3 sample_normal_ws = uncompress_normal_octahedral_32(gs_quad_normal_oct[sample_gs_idx.x][sample_gs_idx.y]);
                const float3 sample_vs_normal = mul(camera.view, float4(sample_normal_ws, 0.0f)).xyz;
                const float  geo_weight     = sample_depth != 0.0f ? surface_weight_dist_limited(inv_half_res_render_target_size, camera.near_plane, center_depth, center_vs, center_vs_normal, sample_vs_data.xyz, sample_vs_normal) : 0.0f;
                // gs_luma_blurred and gs_rgb_blurred are already in log (perceptual) space.
                const float  log_luma = gs_luma_blurred[sample_gs_idx.x][sample_gs_idx.y];
                const float3 log_rgb  = gs_rgb_blurred[sample_gs_idx.x][sample_gs_idx.y];
                partial_sum   += log_luma * geo_weight;
                partial_count += geo_weight;
                rgb_partial_sum += log_rgb * geo_weight;
#if RTGI_USE_QUAD
                partial_rayshortness += gs_quad_rayshortness[sample_gs_idx.x][sample_gs_idx.y] * geo_weight;
                partial_footprint += gs_quad_accept_count[sample_gs_idx.x][sample_gs_idx.y] * geo_weight;
#endif
            }
        }
        gs_luma_partial[flat_quad][quad_lane]  = float2(partial_sum, partial_count);
        gs_rgb_partial[flat_quad][quad_lane]  = rgb_partial_sum;
#if RTGI_USE_QUAD
        gs_rayshortness_partial[flat_quad][quad_lane] = partial_rayshortness;
        gs_footprint_partial[flat_quad][quad_lane] = partial_footprint;
#endif
    }

    GroupMemoryBarrierWithGroupSync();

    {
        const int2 quad_id   = int2(gtid) / 2;
        const int  quad_lane = int(gtid.y & 1u) * 2 + int(gtid.x & 1u);
        const int  flat_quad = quad_id.y * NUM_QUADS_X + quad_id.x;
        const int2 gs_center = quad_id + int2(2, 2);

        if (quad_lane == 0)
        {
            // Center included unconditionally. gs arrays already in log space — read directly.
            float total_sum   = gs_luma_blurred[gs_center.x][gs_center.y];
            float total_count = 1.0f;
            float3 total_rgb_sum = gs_rgb_blurred[gs_center.x][gs_center.y];
            [unroll]
            for (int l = 0; l < 4; l++)
            {
                const float2 p = gs_luma_partial[flat_quad][l];
                total_sum   += p.x;
                total_count += p.y;
                total_rgb_sum += gs_rgb_partial[flat_quad][l];
            }
            gs_quad_luma_mean[flat_quad]  = total_count > 1.0f ? total_sum / total_count : total_sum;
            gs_rgb_mean_result[flat_quad] = perceptual_to_linear(total_rgb_sum / max(total_count, 1.0f));
#if RTGI_USE_QUAD
            {
                float total_rayshortness = gs_quad_rayshortness[gs_center.x][gs_center.y];
                [unroll]
                for (int l = 0; l < 4; l++)
                    total_rayshortness += gs_rayshortness_partial[flat_quad][l];
                gs_quad_rayshortness_result[flat_quad] = total_count > 1.0f ? total_rayshortness / total_count : 0.0f;
            }
            {
                float total_footprint = gs_quad_accept_count[gs_center.x][gs_center.y];
                [unroll]
                for (int l = 0; l < 4; l++)
                    total_footprint += gs_footprint_partial[flat_quad][l];
                gs_quad_footprint_result[flat_quad] = min(total_footprint / 20.0f, 1.0f);
            }
#endif
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Per-thread readout of quad results — visible for the rest of the shader.
    const int2  quad_id_gs   = int2(gtid) / 2;
    const int   flat_quad_gs = quad_id_gs.y * NUM_QUADS_X + quad_id_gs.x;
    const float3 quad_rgb_mean  = gs_rgb_mean_result[flat_quad_gs];
    const float  quad_luma_mean = perceptual_to_linear(gs_quad_luma_mean[flat_quad_gs]);

#if RTGI_USE_QUAD
    // Unweighted 5x5 geometric mean for the firefly ceiling — no surface test.
    // Isolated detail pixels (corners, thin features) would get near-zero samples under a
    // geometry-weighted mean, making firefly suppression ineffective there.
    float ff_luma_log_sum = 0.0f;
    float3 ff_rgb_log_sum = float3(0.0f, 0.0f, 0.0f);
    int ff_count = 0;
    {
        const int2 ff_center = quad_id_gs + int2(2, 2);
        [unroll] for (int fy = -2; fy <= 2; fy++) {
        [unroll] for (int fx = -2; fx <= 2; fx++) {
            const int2 s = ff_center + int2(fx, fy);
            if (gs_quad_vs[s.x][s.y].w != 0.0f) {
                ff_luma_log_sum += gs_luma_blurred[s.x][s.y];
                ff_rgb_log_sum  += gs_rgb_blurred[s.x][s.y];
                ff_count++;
            }
        }}
    }
    const float  firefly_luma_mean = ff_count > 0 ? perceptual_to_linear(ff_luma_log_sum / ff_count) : quad_luma_mean;
    const float3 firefly_rgb_mean  = ff_count > 0 ? perceptual_to_linear(ff_rgb_log_sum  / ff_count) : quad_rgb_mean;
#endif

    // gs_albedo_rgb is fully written — center pixel sits at gtid + TOTAL_FILTER_REACH in the 20x20 tile
    const int2 grad_idx = int2(gtid) + TOTAL_FILTER_REACH;

    // Load Pixel Data
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];

    // const float pixel_samplecnt = push.attach.half_res_samplecnt.get()[clamped_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_normals.get()[clamped_index]);

    // Reconstruct pixel positions based on depth
    const float pixel_vs_depth = linearise_depth(depth, camera.near_plane);
    const float2 uv = (float2(clamped_index.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;
    const float pixel_ws_size = ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, depth);
    const float pixel_ws_size_rcp = rcp(pixel_ws_size);

#if !RTGI_USE_QUAD
    // Geometric mean is used to suppress fireflies
    float y_mean_geometric_acc = 0.0f;
    float3 rgb_mean_geometric_acc = float3(0.0f, 0.0f, 0.0f);
    float valid_geometric_mean_samples = 0.0f;
    // Per-channel log-space sq accumulators for signal variance — same tap set as geometric mean.
    // Max across channels prevents single-channel noise from being masked by smooth luma.
    float3 rgb_log_sq_acc = float3(0.0f, 0.0f, 0.0f);

    // Variance, ray length and valid footprint samples are used for pre blur filter guiding later
    float y_mean_acc = 0.0f;
    float y_variance_acc = 0.0f;
    float y_max = 0.0f;
#if !RTGI_USE_QUAD
    float ray_shortness_acc = 0.0f;
#endif
    float valid_footprint_samples = 0.0f;

    float3 albedo_mean_acc = float3(0.0f, 0.0f, 0.0f);

    // Per-pixel albedo gradient estimation: accumulate |neighbor_alb - center_alb| from groupshared
    const float3 pp_center_alb = unpack_rgb(gs_albedo_rgb[grad_idx.x][grad_idx.y]);
    float pp_alb_diff_acc = 0.0f;
    float pp_alb_diff_count = 0.0f;
    // Coefficient of variation accumulators (same 3x3 window)
    float pp_alb_lum_acc = 0.0f;
    float pp_alb_lum_sq_acc = 0.0f;

    // The raw signal is pre blurred in a star shape with 5 taps to conserve energy from the firefly filter.
#if !RTGI_USE_QUAD
    float4 star_blurred_diffuse_acc = float4(0,0,0,0);
    float2 star_blurred_diffuse2_acc = float2(0,0);
    float star_blurred_weight_acc = 0.0f;
#endif

#if !RTGI_USE_QUAD
    const float max_visibility_raylen = pixel_ws_size * MAX_VISIBILITY_PIXEL_RANGE;
#endif

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
#if !RTGI_USE_QUAD
            const float sample_ray_length = push.attach.ray_length_image.get()[load_idx];   // preloading via shared mem reduces perf, distribute loading onto multiply hw units
#endif
            const float4 preload_v = gs_sample_value_vs[preload_index.x][preload_index.y];
            const float3 sample_value_vs = preload_v.xyz;
            const float sample_depth = preload_v.w;
            float sample_y = sample_diffuse.w;
            sample_y = max(sample_y, 1e-3f);          // Values below this start to break float16 precision, have to clamp radiance up to that value as a minimum for statistic analysis.
            const bool is_sky = sample_depth == 0.0f;
            if (is_sky)
            {
                continue;
            }

            if (!is_part_of_center_blur(int2(x,y)))
            {
                float geometric_mean_acc_value = linear_to_perceptual(sample_y);
                y_mean_geometric_acc += geometric_mean_acc_value;
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
                const float plane_distance = planar_surface_distance(vs_position, vs_normal, sample_value_vs) * pixel_ws_size_rcp;
                const float geometric_weight = abs(plane_distance) < 1.0f ? 1.0f : 0.0f;

#if !RTGI_USE_QUAD
                if (is_part_of_center_blur(int2(x,y)))
                {
                    star_blurred_weight_acc += geometric_weight;
                    star_blurred_diffuse_acc += geometric_weight * sample_diffuse;
                    star_blurred_diffuse2_acc += geometric_weight * sample_diffuse2;
                }
#endif

                y_mean_acc += sample_y * geometric_weight;
                y_variance_acc += square(sample_y) * geometric_weight;
                y_max = max(y_max, sample_y) * geometric_weight;
#if !RTGI_USE_QUAD
                // square the relative ray length (ranges from 0-1) to promote small values in sum. Later we take sqrt(sqrt(acc*weight_sum)) to get a scaled average of ray shortness.
                ray_shortness_acc += ((1.0f - ((min(1.0f, sample_ray_length * rcp(max_visibility_raylen)))))) * geometric_weight;
#endif

                const float3 sample_albedo = push.attach.view_cam_half_res_albedo.get()[load_idx].rgb;
                albedo_mean_acc += sample_albedo * geometric_weight;

                valid_footprint_samples += geometric_weight;

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
#endif // !RTGI_USE_QUAD

    let rtgi = push.attach.globals->rtgi_settings;

    // --- Footprint Quality ---
    float footprint_quality = 1.0f;
    {
#if RTGI_USE_QUAD
        footprint_quality = gs_quad_footprint_result[flat_quad_gs];
#else
        footprint_quality = valid_footprint_samples * rcp(float(INNER_FILTER_TAPS_TOTAL));
#endif
    }

#if !RTGI_USE_QUAD
    // --- Geometric means (used by both guide output switch and firefly filter) ---
    const float y_mean_geometric = perceptual_to_linear(y_mean_geometric_acc * rcp(valid_geometric_mean_samples));
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
    //const float  out_cov_guide = quad_cov; // cov cut off, hardcoded to 1.0f in write
    const float  out_std_dev   = 0.0f; // cov cut off
    const float  out_mean      = quad_luma_mean;
    const float3 out_rgb_mean  = quad_rgb_mean;
    const float  dbg_geo_mean  = quad_luma_mean;
    //const float  dbg_geo_var   = quad_variance; // cov cut off
    //const float  dbg_cov       = quad_cov; // cov cut off
#else
    const float  out_cov_guide = y_log_std_dev;
    const float  out_std_dev   = y_log_std_dev;
    const float  out_mean      = y_mean_geometric;
    const float3 out_rgb_mean  = rgb_mean_geometric;
    const float  dbg_geo_mean  = y_mean_geometric;
    const float  dbg_geo_var   = max(max(rgb_log_variance.r, rgb_log_variance.g), rgb_log_variance.b);
    const float  dbg_cov       = y_log_std_dev;
#endif

    // --- Firefly Filter ---
    float4 filtered_diffuse;
    float2 filtered_diffuse2;
    float firefly_energy_factor = 1.0f;
#if !RTGI_USE_QUAD
    float y_std_dev;
    float y_luma_mean;
    float y_mean_geometric_over_linear = 1.0f;
#endif
#if RTGI_USE_QUAD
    const int2  quad_center_cell = quad_id_gs + int2(2, 2);
    const float3 quad_rep_normal_ws = uncompress_normal_octahedral_32(gs_quad_normal_oct[quad_center_cell.x][quad_center_cell.y]);
    const float3 quad_rep_vs        = gs_quad_vs[quad_center_cell.x][quad_center_cell.y].xyz;
    const float3 quad_rep_normal_vs = mul(camera.view, float4(quad_rep_normal_ws, 0.0f)).xyz;
    const float  quad_blend         = depth != 0.0f ? surface_weight_dist_limited(inv_half_res_render_target_size, camera.near_plane, depth, vs_position, vs_normal, quad_rep_vs, quad_rep_normal_vs) : 0.0f;
    const bool   pixel_matches_quad = quad_blend > 0.0f;
#endif
    {
        const float EPSILON = 1e-8f;
#if RTGI_USE_QUAD
        const int2 center_blur_cell = quad_center_cell;
        const float4 quad_star_diffuse  = gs_quad_diffuse[center_blur_cell.x][center_blur_cell.y];
        const float2 quad_star_diffuse2 = gs_quad_diffuse2[center_blur_cell.x][center_blur_cell.y];
        const float4 raw_diffuse  = push.attach.diffuse_raw.get()[clamped_index];
        const float2 raw_diffuse2 = push.attach.diffuse2_raw.get()[clamped_index].rg;
        const float4 star_blurred_diffuse  = lerp(raw_diffuse,  quad_star_diffuse,  quad_blend);
        const float2 star_blurred_diffuse2 = lerp(raw_diffuse2, quad_star_diffuse2, quad_blend);
#else
        const float4 star_blurred_diffuse = star_blurred_diffuse_acc * rcp(star_blurred_weight_acc + EPSILON);
        const float2 star_blurred_diffuse2 = star_blurred_diffuse2_acc * rcp(star_blurred_weight_acc + EPSILON);
#endif

        const bool use_star_blur = rtgi.firefly_filter_enabled != 0 && rtgi.firefly_star_blur_enabled != 0;
        if (use_star_blur)
        {
            filtered_diffuse = star_blurred_diffuse;
            filtered_diffuse2 = star_blurred_diffuse2;
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

        const float ceiling_factor = max(1.0f, rtgi.firefly_filter_ceiling * footprint_quality);

#if !RTGI_USE_QUAD
        y_mean_geometric_over_linear = out_mean / y_luma_mean;
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
        const float  geo_mean_for_ceiling     = firefly_luma_mean;
        const float3 rgb_mean_for_ceiling     = firefly_rgb_mean;
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
    const float ray_shortness_mean = sqrt(sqrt(gs_quad_rayshortness_result[flat_quad_gs]));
#else
    const float ray_shortness_mean = sqrt(sqrt(ray_shortness_acc * rcp(valid_footprint_samples)));
#endif
    const float raylength_filter_guide = (1.0f - ray_shortness_mean) * footprint_quality;

    // Pixels that don't belong to their quad's representative surface get -1 mean
    // so the pre-blur can skip variance weighting for them.
    float written_mean = out_mean;
#if RTGI_USE_QUAD
    if (!pixel_matches_quad)
        written_mean = -1.0f;
#endif


    debug_image_tile_draw(dbg, 0,  dtid, float4(TurboColormap(log(dbg_geo_mean+1) * 0.13), 2.0f), 2);
    // debug_image_tile_draw(dbg, 0,  dtid, float4(TurboColormap(footprint_quality), 2.0f), 2);
    // debug_image_tile_draw(dbg, -1, dtid, lerp(float4(1,0,0,2), float4(0,1,0,2), quad_blend), 2);
    //if (written_mean == -1.0f) debug_image_tile_draw(dbg, -1, dtid, float4(1, 0, 0, 2), 2);

    push.attach.pre_filtered_diffuse_image.get()[dtid] = filtered_diffuse;
    push.attach.pre_filtered_diffuse2_image.get()[dtid] = filtered_diffuse2;
    push.attach.firefly_factor_image.get()[dtid] = firefly_energy_factor;
    push.attach.spatial_std_dev_image.get()[dtid] = float2(out_std_dev, written_mean);
    push.attach.filter_guide_image.get()[dtid] = pack_filter_guide(raylength_filter_guide);
}