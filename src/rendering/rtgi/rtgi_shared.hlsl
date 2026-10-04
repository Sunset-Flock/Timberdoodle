#pragma once

#include "shader_shared/rtgi.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

#define RTGI_QUAD_FILTER_EXTENT 2  // valid: 1 (3x3), 2 (5x5)
#define RTGI_QUAD_FILTER_STRIDE 1  // cell spacing of the outer gather; >1 widens reach at same tap count (e.g. extent 1 + stride 2)

#define RTGI_RADIANCE_SCALE (1e4f)

// == Packed history sample counters ===========================================
// The normal (slow) temporal sample count and the fast-history frame count are packed into a single
// R16_UINT texel:
//   - normal count : low 10 bits, piecewise encoded so ray-weighted history (RTGI_RAY_WEIGHTED_HISTORY)
//                    can exceed max_temporal_samples:
//                      code [0, 511]    -> count = code * 0.25              ([0, 127.75], 0.25 steps)
//                      code [512, 1022] -> count = 128 + (code - 512) * 2   ([128, 1148], 2.0 steps)
//   - fast count   : range [0, 15],  0.25 steps -> 0..60,   high 6 bits.
// The code 0x3FF (1023) in the 10-bit normal field is reserved as the SKY sentinel (unpacks to -1),
// matching the old float image's <0 sky marker that the reproject/accumulate passes early-out on. The
// normal count is clamped to RTGI_COUNT_NORMAL_MAX (code 1022) so a real count never hits the sentinel.
static const uint RTGI_COUNT_NORMAL_MASK = 0x3FFu;  // 10 bits
static const uint RTGI_COUNT_FAST_MASK   = 0x3Fu;   // 6 bits
static const uint RTGI_COUNT_SKY         = 0x3FFu;  // sentinel in the normal field
static const uint RTGI_COUNT_COARSE_CODE = 512u;    // first code of the coarse (2.0 step) range
static const float RTGI_COUNT_COARSE_BASE = 128.0f; // count at RTGI_COUNT_COARSE_CODE
static const float RTGI_COUNT_NORMAL_MAX = 1148.0f; // highest real normal count (code 1022)

func rtgi_pack_sample_counts(float normal_count, float fast_count) -> uint
{
    const float c = clamp(normal_count, 0.0f, RTGI_COUNT_NORMAL_MAX);
    const uint n = c < RTGI_COUNT_COARSE_BASE - 0.125f
        ? uint(round(c * 4.0f))                                                            // 0..511
        : RTGI_COUNT_COARSE_CODE + uint(round((c - RTGI_COUNT_COARSE_BASE) * 0.5f));       // 512..1022
    const uint f = uint(round(clamp(fast_count,   0.0f, 15.0f)                 * 4.0f)); // 0..60
    return (n & RTGI_COUNT_NORMAL_MASK) | ((f & RTGI_COUNT_FAST_MASK) << 10u);
}

// Sky sentinel: normal field flags sky (unpacks to -1), fast field 0.
func rtgi_pack_sample_counts_sky() -> uint
{
    return RTGI_COUNT_SKY;
}

func rtgi_unpack_normal_count(uint packed) -> float
{
    const uint n = packed & RTGI_COUNT_NORMAL_MASK;
    if (n == RTGI_COUNT_SKY) return -1.0f;
    return n < RTGI_COUNT_COARSE_CODE
        ? float(n) * 0.25f
        : RTGI_COUNT_COARSE_BASE + float(n - RTGI_COUNT_COARSE_CODE) * 2.0f;
}

// Ray-weighted temporal history.
//   0 = history count capped at max_temporal_samples RAYS. Once capped, a pixel that shoots k rays gets
//       blend weight k/(1+cap) but also evicts k rays of history, so extra rays only shorten the time
//       window and do NOT lower the noise.
//   1 = history count is only capped in TIME. Up to max_temporal_samples it grows linearly exactly as
//       before (bit-identical at 1 ray/frame). Above it, the history decays by 1/max_temporal_samples per
//       frame instead of being clamped, so every sample's weight decays with its AGE in frames and the
//       count settles at k * max_temporal_samples for a steady k rays/frame. An 8-ray frame then counts 8x
//       a 1-ray frame, and a pixel traced at 8 rays/frame holds 8x the samples (8x lower variance) at
//       the same time window. Lag in frames is unchanged, and the fast-history confidence scaling stays
//       consistent because blend ~ k / count is still 1 / (window in frames).
#define RTGI_RAY_WEIGHTED_HISTORY 1

// History cap the reproject pass clamps the carried count to.
func rtgi_history_count_cap(float max_temporal_samples) -> float
{
#if RTGI_RAY_WEIGHTED_HISTORY
    return RTGI_COUNT_NORMAL_MAX;
#else
    return max_temporal_samples;
#endif
}

// Accumulated history sample count after integrating `rays` fresh samples on top of the reprojected
// `carry`. rays == 0 (no-ray pixel) keeps the carry below the cap, like before.
func rtgi_accumulate_sample_count(float carry, float rays, float max_temporal_samples) -> float
{
#if RTGI_RAY_WEIGHTED_HISTORY
    const float grown  = carry + rays;                                           // below the cap: plain growth
    const float leaked = carry * (1.0f - rcp(max(max_temporal_samples, 1.0f))) + rays; // above: age decay
    return min(min(grown, max(max_temporal_samples, leaked)), RTGI_COUNT_NORMAL_MAX);
#else
    return min(max_temporal_samples, carry + rays);
#endif
}

func rtgi_unpack_fast_count(uint packed) -> float
{
    return float((packed >> 10u) & RTGI_COUNT_FAST_MASK) * 0.25f;
}

// Converts a stored reproject_corner (see rtgi_temporal.hlsl's rtgi_reproject_corner) into the UV to
// Gather the reprojected 2x2 history block from. Used by the accumulate pass to read the reprojection
// metadata written this frame by the reproject pass.
func rtgi_reproject_gather_uv(uint2 corner_plus_one, float2 inv_half_res_render_target_size) -> float2
{
    return float2(corner_plus_one) * inv_half_res_render_target_size; // == (origin + 1) * inv_size
}

// == Half-res depth encoding ==================================================
// The half-res depth image holds linear view depth (written by gen_gbuffer). Sky is 0.
// Always read it through these helpers.

// Half-res depth image value -> world position. Undefined for sky (0).
func rtgi_half_res_depth_to_world_space(CameraInfo camera, float2 ndc_xy, float half_res_depth) -> float3
{
    return linear_depth_to_world_space(camera, ndc_xy, half_res_depth);
}

// Half-res depth image value -> view position, NEGATED (same convention as rtgi_upscale's -inv_proj unproject).
func rtgi_half_res_depth_to_neg_view_space(CameraInfo camera, float2 ndc_xy, float half_res_depth) -> float3
{
    return -camera_view_ray_vs(camera, ndc_xy) * half_res_depth;
}

// calc_pixel_width_ws for a half-res depth image value. Exact without the divide: near / ndc_depth == linear depth.
func rtgi_half_res_pixel_width_ws(float2 inv_render_target_size, float near_plane, float half_res_depth) -> float
{
    return inv_render_target_size.y * 2.0f * half_res_depth;
}

struct PixelData
{
    float2 uv;
    float2 ndc;
    float  depth_vs; // linear view-space depth (distance along the view direction), 0 == sky
    float3 position_ws;
    float3 position_vs;
    float3 normal_ws;
    float3 normal_vs;
};

func calc_pixel_data(
    uint2 dtid,
    float2 inv_half_res_render_target_size,
    const CameraInfo camera,
    Texture2D<float> depth_tex,
    Texture2D<uint> normals_tex) -> PixelData
{
    PixelData pd;
    pd.uv          = (float2(dtid) + 0.5f) * inv_half_res_render_target_size;
    pd.ndc         = pd.uv * 2.0f - 1.0f;
    pd.depth_vs    = depth_tex[dtid];
    pd.position_ws = rtgi_half_res_depth_to_world_space(camera, pd.ndc, pd.depth_vs);
    pd.position_vs = mul(camera.view, float4(pd.position_ws, 1.0f)).xyz;
    pd.normal_ws   = uncompress_normal_octahedral_32(normals_tex[dtid]);
    pd.normal_vs   = mul(camera.view, float4(pd.normal_ws, 0.0f)).xyz;
    return pd;
}

// Converts a ray hit distance to a bounded [0,1] "shortness": 1 for a coincident hit, ramping linearly
// to 0 at the max visibility range (max_visibility_pixel_range = RtgiSettings.max_visibility_pixel_range,
// in half-res pixel widths). The trace / blend passes store the per-pixel MEAN of this over all of a
// pixel's rays into the ray-length texture, so the denoiser guide reads a stable, bounded signal instead
// of raw (unbounded, single-ray, high-variance) hit distances.
func calc_ray_shortness(float ray_length, float pixel_width_ws, float max_visibility_pixel_range) -> float
{
    const float max_visibility_raylen = pixel_width_ws * max_visibility_pixel_range;
    return 1.0f - min(1.0f, square(ray_length * rcp(max(max_visibility_raylen, 1e-8f))));
}

// Relative perceived brightness of the three color channels, normalized so the brightest (green) is 1.
// Derived from the Rec.709 luma weights (0.2126, 0.7152, 0.0722) divided by the green weight.
static const float3 RTGI_CHANNEL_PERCEIVED_BRIGHTNESS = float3(0.2973f, 1.0f, 0.1009f);

// Same relationship expressed in perceptual (natural-log) space relative to green: ln(brightness / green).
// Green is 0; the others are negative because red and blue are perceived dimmer than green. Natural log so
// it can be added directly to the ln-based perceptual values (linear_to_perceptual / perceptual_to_linear).
static const float3 RTGI_CHANNEL_PERCEIVED_BRIGHTNESS_LN = float3(-1.2129f, 0.0f, -2.2936f);

// Lower clamp for a radiance value BEFORE taking its log, when accumulating a geometric (log) mean.
// Human brightness perception is logarithmic (Weber-Fechner law), so below a certain luminance — scaled
// by the current exposure — darker values are perceptually indistinguishable from black. Without a floor,
// log(radiance) explodes toward large negative values for tiny inputs and drags the log-radiance mean far below
// anything a viewer could perceive, distorting the geometric mean. This returns that perceptual floor:
// the darkest radiance still meaningfully distinguishable under `exposure` (the pre-tonemap multiplier,
// globals.exposure). Brighter scenes (smaller exposure) raise the floor. Max your radiance with this before
// log().
func calc_perceptual_radiance_floor(float inv_exposure) -> float
{
    return inv_exposure * RTGI_RADIANCE_SCALE * 1e-3f;
}

func linear_to_perceptual(float v, float inv_exposure) -> float
{
    return log(max(v, calc_perceptual_radiance_floor(inv_exposure)));
}

func linear_to_perceptual_rgb(float3 v, float inv_exposure) -> float3
{
    return log(max(v, calc_perceptual_radiance_floor(inv_exposure)));
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

// Perceptual (log-space) radiance inferred from the stored perceptual (log) rgb — the weighted geometric-mean
// radiance (log of r^0.25 * g^0.5 * b^0.25). Lets us drop a dedicated log-radiance channel and reconstruct it from
// the rgb channels, freeing that texture slot (used to carry ray shortness instead).
func perceptual_radiance_from_rgb(float3 perceptual_rgb) -> float
{
    return dot(perceptual_rgb, float3(0.25f, 0.5f, 0.25f));
}

func calc_pixel_width_ws(float2 inv_render_target_size, float near_plane, float depth) -> float
{
    // The further away the pixel is, the larger difference we allow.
    // The scale is proportional to the size the pixel takes up in world space.
    const float pixel_size_on_near_plane = inv_render_target_size.y;
    const float near_plane_ws_size = near_plane * 2;
    const float pixel_width_ws = pixel_size_on_near_plane * near_plane_ws_size * rcp(depth + 0.0000001f);
    return pixel_width_ws;
}

// Fixed uniform ray count per pixel used when ray redistribution is disabled: exactly
// max(floor(ray_budget), 1) rays for every geometry pixel, in both the repacked and classic paths.
// Hard ray budget of the redistributing (repacked) path: rays per frame, never exceeded. The distribute pass
// drains demand to it and additionally clamps the ray list to it; the list trace never reads past it.
func rtgi_hard_ray_budget(uint2 half_res_size, RtgiSettings settings) -> uint
{
    const float ray_pct = clamp(settings.ray_percentage, 0.0f, float(RTGI_RAY_LIST_CAPACITY_MUL));
    const float min_budget = clamp(settings.min_ray_budget, 0.0f, 1.0f);
    return uint(float(half_res_size.x * half_res_size.y) * max(ray_pct, min_budget));
}

// Ray list entries the list trace may read this frame: the allocated count, never past the list capacity and, when
// redistributing, never past the hard budget.
func rtgi_ray_list_limit(uint2 half_res_size, RtgiSettings settings) -> uint
{
    const uint capacity = half_res_size.x * half_res_size.y * RTGI_RAY_LIST_CAPACITY_MUL;
    return settings.use_ray_redistribution != 0 ? min(capacity, rtgi_hard_ray_budget(half_res_size, settings)) : capacity;
}

func calc_fixed_rays_per_pixel(float ray_percentage) -> uint
{
    return max(uint(floor(max(ray_percentage, 0.0f))), 1u);
}

// Redistribution off: fixed rays per geometry pixel, max(floor(ray_budget), 1 per active signal), split evenly
// between diffuse and specular (diffuse keeps the odd ray). Every active signal always gets its base ray, like
// with redistribution; below 2 rays/pixel with specular on that means slightly more rays than the budget.
func calc_fixed_ray_split(float ray_percentage, bool specular_active, out uint total, out uint specular)
{
    total = max(calc_fixed_rays_per_pixel(ray_percentage), specular_active ? 2u : 1u);
    specular = specular_active ? total / 2u : 0u;
}

// Extra rays (beyond the mandatory base ray) a geometry pixel wants, given its reprojected history count, as a
// fraction of rays (the diffuse / specular share multiplies it before rounding). Exponential over the convergence
// target T (x = count / T): amplitude exp(-4x), `amplitude` extras at x = 0, 0 at x >= 1 (fast early drop, weak
// tail). The amplitude is sized to the per-pixel ray cap (rtgi_calc_ray_demand), so the curve's shape survives the
// cap instead of being flattened by it. amplitude 5 examples (x = 0 / 0.125 / 0.25 / 0.5 / 0.75): 5 / 3.0 / 1.8 / 0.7 / 0.25.
func calc_desired_extra_rays(float reproj_sample_count, float target, float amplitude) -> float
{
    if (reproj_sample_count >= target || target <= 0.0f) { return 0.0f; }
    const float x = max(reproj_sample_count, 0.0f) / target;
    return amplitude * exp(-4.0f * x);
}

// == Single ray budget: diffuse + specular ======================================
// Every traced ray is EITHER diffuse or specular, and both kinds come out of the same frame budget. A pixel
// wants 1 base ray per signal plus deficit-proportional extras per signal (each signal's own history count vs
// the convergence target). The reproject pass totals this demand, the distribute pass drains it, and a pixel's
// allocated extras are split between the two signals in proportion to their deficits. In the ray list a
// pixel's entries are [diffuse 0..n_d) followed by [specular 0..n_s).
struct RtgiRayDemand
{
    uint diffuse;  // desired diffuse rays (1 + extras), 0 for sky
    uint specular; // desired specular rays (1 + extras), 0 for sky or specular disabled
};

// == Ray demand: diffuse / specular share ==========================================
// Each signal's EXTRA rays (deficit-driven, see rtgi_calc_ray_demand) are scaled by a per-pixel factor computed
// once by the reproject pass (rtgi_calc_ray_share in rtgi_temporal.hlsl) and stored in the ray_impact image,
// which every ray demand caller (reproject, distribute, classic trace) reads. The base ray per signal stays.
struct RtgiRayImpact
{
    float diffuse;
    float specular;
};

// Max rays (diffuse + specular, base included) one pixel may request per frame (settings.max_rays_per_pixel, never
// below the base rays). Many rays in ONE frame share the
// same frame's noise pattern / guide and are too temporally similar: a few pixels getting ~32 rays at once form
// visible stripes. The demand curve's amplitude is sized so a fully fresh pixel asks for exactly the cap: the
// share factors sum to 2 with specular on (1 with it off), so each signal's curve peaks at extra_cap / that sum.
// Above the cap (only reachable by rounding) both signals' extras are scaled down proportionally.
func rtgi_calc_ray_demand(float diffuse_sample_count, float specular_sample_count, RtgiRayImpact impact, RtgiSettings settings) -> RtgiRayDemand
{
    const bool specular_on = settings.specular_enabled != 0;
    const uint base_rays = specular_on ? 2u : 1u;
    const uint max_rays = max(uint(max(settings.max_rays_per_pixel, 0)), base_rays);
    const float base = float(base_rays);
    const float extra_cap = float(max_rays) - base;
    const float amplitude = extra_cap / (specular_on ? 2.0f : 1.0f);
    const float diffuse_extra = calc_desired_extra_rays(diffuse_sample_count, settings.fast_convergence_samples, amplitude) * impact.diffuse;
    // The specular history is capped at specular_max_temporal_frames samples, so never ask for more than that.
    const float specular_target = min(settings.specular_fast_convergence_samples, settings.specular_max_temporal_frames);
    const float specular_extra = specular_on ? calc_desired_extra_rays(specular_sample_count, specular_target, amplitude) * impact.specular : 0.0f;
    const float extra_scale = min(1.0f, extra_cap / max(diffuse_extra + specular_extra, 1e-6f));
    RtgiRayDemand d;
    d.diffuse = 1u + uint(diffuse_extra * extra_scale + 0.5f);
    d.specular = specular_on ? 1u + uint(specular_extra * extra_scale + 0.5f) : 0u;
    // Rounding both halves up can exceed the cap by one: take it from the larger request.
    if (d.diffuse + d.specular > max_rays)
    {
        if (d.diffuse >= d.specular) { d.diffuse -= 1u; } else { d.specular -= 1u; }
    }
    return d;
}

func rtgi_ray_demand_base(RtgiRayDemand d) -> uint
{
    return (d.diffuse > 0u ? 1u : 0u) + (d.specular > 0u ? 1u : 0u);
}

// Splits `extra` allocated extra rays between the signals by their extra demands (rounded, each capped).
func rtgi_split_extra_rays(uint extra, RtgiRayDemand d, out uint extra_diffuse, out uint extra_specular)
{
    const uint diffuse_extra_demand  = d.diffuse  > 0u ? d.diffuse  - 1u : 0u;
    const uint specular_extra_demand = d.specular > 0u ? d.specular - 1u : 0u;
    const uint total_demand = diffuse_extra_demand + specular_extra_demand;
    extra_specular = total_demand > 0u ? min((extra * specular_extra_demand + total_demand / 2u) / total_demand, specular_extra_demand) : 0u;
    extra_diffuse  = extra - extra_specular;
}

func calc_plane_distance(float3 a_pos, float3 a_norm, float3 b_pos) -> float
{
    return dot(a_pos - b_pos, a_norm);
}

func calc_similar_surface_weight(const float rcp_pixel_width_ws, const float3 a_pos, const float3 a_norm, const float3 b_pos, const float3 b_norm, const float px_threshold = 1.2f) -> float
{
    const float within_acceptable_plane_dist_a = 1.0f - saturate(calc_plane_distance(a_pos, a_norm, b_pos) * rcp(px_threshold) * rcp_pixel_width_ws);
    const float within_acceptable_plane_dist_b = 1.0f - saturate(calc_plane_distance(b_pos, b_norm, a_pos) * rcp(px_threshold) * rcp_pixel_width_ws);
    return within_acceptable_plane_dist_a * within_acceptable_plane_dist_b;
}

// Same as surface_similarity but also rejects pairs of points that share a plane yet are
// far apart in world space — guards against accidental coplanar matches across large distances.
func calc_similar_surface_weight_dist_limited(const float rcp_pixel_width_ws, const float3 a_pos, const float3 a_norm, const float3 b_pos, const float3 b_norm, const float px_threshold = 1.2f) -> float
{
    const float surface_weight              = calc_similar_surface_weight(rcp_pixel_width_ws, a_pos, a_norm, b_pos, b_norm, px_threshold);
    const float dist_in_pixel_widths        = abs(distance(a_pos, b_pos)) * rcp_pixel_width_ws;
    const float pixel_widths_threshold      = 32.0f * px_threshold;
    return step(dist_in_pixel_widths, pixel_widths_threshold) * surface_weight;
}

// Due to the low resolution the tracing and de-noising runs at we have to use normals only as a strong suggestion, not for cutoff.
func calc_similar_normal_weight(float3 normal, float3 other_normal) -> float
{
    const float validity = (max(0.1f, dot(normal, other_normal) + 0.85f)) * (1.0f / 1.85f);
    const float tight_validity = (square(validity));
    return tight_validity;
}

func calc_perceptual_difference_weight(float a_radiance_perceptual, float b_radiance_perceptual, float tolerance) -> float
{
    const float perceptual_difference = a_radiance_perceptual - b_radiance_perceptual;
    return exp(-square(2.0f * perceptual_difference * rcp(tolerance)));
    // return rcp(square(perceptual_difference * 8 * rcp(tolerance) + 1));
}

static const float3 g_Poisson8[8] =
{
    float3( -0.4706069, -0.4427112, +0.6461146 ),
    float3( -0.9057375, +0.3003471, +0.9542373 ),
    float3( -0.3487388, +0.4037880, +0.5335386 ),
    float3( +0.1023042, +0.6439373, +0.6520134 ),
    float3( +0.5699277, +0.3513750, +0.6695386 ),
    float3( +0.2939128, -0.1131226, +0.3149309 ),
    float3( +0.7836658, -0.4208784, +0.8895339 ),
    float3( +0.1564120, -0.8198990, +0.8346850 )
};

static const float3 g_Poisson16[16] =
{
    float3( -0.8471255, -0.2785693, +0.8917524 ),
    float3( -0.2297938, -0.2639703, +0.3499793 ),
    float3( -0.5808771, -0.7114083, +0.9184334 ),
    float3( -0.1064163, -0.8944703, +0.9007783 ),
    float3( +0.4220688, -0.8510922, +0.9500000 ),
    float3( +0.2222625, -0.4196452, +0.4748712 ),
    float3( +0.7828983, -0.4547251, +0.9053754 ),
    float3( +0.9250575, +0.0424033, +0.9260288 ),
    float3( +0.3838071, +0.0451333, +0.3864517 ),
    float3( +0.7267400, +0.5220043, +0.8947846 ),
    float3( +0.3465116, +0.8506802, +0.9185462 ),
    float3( +0.0405502, +0.3929285, +0.3950154 ),
    float3( -0.1696051, +0.9010239, +0.9168478 ),
    float3( -0.6187802, +0.6714641, +0.9131008 ),
    float3( -0.4148493, +0.1812793, +0.4527274 ),
    float3( -0.9135795, +0.2290230, +0.9418487 )
};

float calc_gaussian_weight( float r )
{
    return exp( -0.66f * square(r * 2.71828182846f * 0.5f) ); // assuming r is normalized to 1
}


float3 linear_to_y_co_cg( float3 color )
{
    float y = dot( color, float3( 0.25, 0.5, 0.25 ) );
    float Co = dot( color, float3( 0.5, 0.0, -0.5 ) );
    float Cg = dot( color, float3( -0.25, 0.5, -0.25 ) );

    return float3( y, Co, Cg );
}

float3 y_co_cg_to_linear( float3 color )
{
    float t = color.x - color.z;

    float3 r;
    r.y = color.x + color.z;
    r.x = t + color.y;
    r.z = t - color.y;

    return max( r, 0.0 );
}

float y_co_cg_to_brightness(float3 yCoCg)
{
    const float3 linear = y_co_cg_to_linear(yCoCg);
    return linear.r + linear.g * 2.0f + linear.b * 0.5f;
}

float rgb_brightness(float3 linear)
{
    return linear.r + linear.g * 2.0f + linear.b * 0.5f;
}

float3 y_co_cg_to_linear_corrected( float y, float sh_y_0, float2 co_cg )
{
    y = max( y, 0.0 );
    co_cg *= ( y + 1e-6 ) / ( sh_y_0 + 1e-6 );

    return y_co_cg_to_linear( float3( y, co_cg ) );
}


float3 sh_resolve_diffuse( float4 sh_y, float2 co_cg, float3 normal )
{
    float y = max(dot( normal, sh_y.xyz ) + 0.5f * sh_y.w, sh_y.w * 0.1f);
    return y_co_cg_to_linear_corrected( y, sh_y.w, co_cg );
}

float4 y_to_sh(float y, float3 direction)
{
    float sh0 = y;
    float3 sh1 = direction * y;
    return float4(sh1, sh0);
}

void radiance_to_y_co_cg_sh(float3 radiance, float3 direction, out float4 sh_y, out float2 co_cg)
{
    float3 y_co_cg = linear_to_y_co_cg(radiance);
    co_cg = y_co_cg.gb;

    sh_y = y_to_sh(y_co_cg.x, direction);
}

// == Specular helpers =========================================================
// The specular channel is linear rgb (RTGI_RADIANCE_SCALE scaled) + hit distance in .a, carried through every
// RTGI pass next to the diffuse SH. These helpers keep the lobe-dependent weights consistent between passes.

// Specular hit distances are stored in f16 images. Misses (huge t) clamp here, which still reprojects like
// "infinitely far" for any practical camera motion.
static const float RTGI_SPECULAR_MAX_HIT_DISTANCE = 1000.0f;

// Approximate tangent of the GGX lobe half-angle (most of the reflected energy lies inside it).
func rtgi_specular_lobe_tan(float roughness) -> float
{
    const float a = roughness * roughness;
    return a;
}

// Screen-space blur radius (in half-res pixels) matching the reflected lobe footprint: the lobe widens with
// distance to the hit, so contact reflections stay sharp and far reflections get the full radius.
func rtgi_specular_blur_radius_px(float roughness, float hit_distance, float pixel_width_ws, float max_radius_px) -> float
{
    const float footprint_ws = min(hit_distance, RTGI_SPECULAR_MAX_HIT_DISTANCE) * rtgi_specular_lobe_tan(roughness);
    return min(footprint_ws * rcp(max(pixel_width_ws, 1e-8f)), max_radius_px);
}

// Detail-normal similarity for specular. Mirrors need almost identical normals to share reflections, rough
// lobes are wide and tolerate a lot.
func rtgi_specular_normal_weight(float3 n_center, float3 n_sample, float roughness) -> float
{
    const float exponent = lerp(512.0f, 8.0f, saturate(roughness * 2.0f));
    return pow(saturate(dot(n_center, n_sample)), exponent);
}

func rtgi_specular_roughness_weight(float roughness_center, float roughness_sample) -> float
{
    return exp(-abs(roughness_center - roughness_sample) * 10.0f);
}

// NRD GetSpecularDominantFactor: how far the GGX lobe's dominant direction leans from N towards R.
func rtgi_spec_dominant_factor(float NoV, float roughness) -> float
{
    const float a = 0.298475f * log(39.4115f - 39.0029f * saturate(roughness));
    return saturate(pow(saturate(1.0f - NoV), 10.8649f) * (1.0f - a) + a);
}

func rtgi_spec_dominant_direction(float3 N, float3 V, float roughness) -> float3
{
    const float3 R = reflect(-V, N);
    return normalize(lerp(N, R, rtgi_spec_dominant_factor(abs(dot(N, V)), roughness)));
}

// log(k / sinh(k)) == log(4 pi C(k)) where C(k) = k / (4 pi sinh k) normalizes a vMF lobe; stable for any k >= 0.
func rtgi_log_vmf_norm(float k) -> float
{
    if (k < 1e-4f) { return 0.0f; }
    return log(k) - (k + log(1.0f - exp(-2.0f * k)) - log(2.0f));
}

// Debug colormap for a (specular) hit distance: log scale, 0 m -> 0, ~1 km -> 1.
func rtgi_hit_distance_debug_color(float hit_distance) -> float3
{
    return Heatmap(saturate(log2(1.0f + max(hit_distance, 0.0f)) * 0.1f));
}

// Converts a perceptual-space radiance value to a display-ready linear color for debug visualization:
// maps back to linear, applies exposure, and scales by debug_visualization_scale.
func perceptual_radiance_colormap(float perceptual, float exposure) -> float3
{
    return Heatmap(perceptual_to_linear(perceptual) * exposure * 0.0003f);
}