#pragma once

#include "shader_shared/rtgi.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"

#define RTGI_QUAD_FILTER_EXTENT 2  // valid: 1 (3x3), 2 (5x5)
#define RTGI_QUAD_FILTER_STRIDE 1  // cell spacing of the outer gather; >1 widens reach at same tap count (e.g. extent 1 + stride 2)

#define VALUE_MULTIPLIER (1e4f)

// == Packed history sample counters ===========================================
// The normal (slow) temporal sample count and the fast-history frame count are packed into a single
// R16_UINT texel. Both keep 0.25 fractional resolution (quantized in quarter steps):
//   - normal count : range [0, 254], 0.25 steps -> 0..1016, low 10 bits.
//   - fast count   : range [0, 15],  0.25 steps -> 0..60,   high 6 bits.
// The value 0x3FF (1020) in the 10-bit normal field is reserved as the SKY sentinel (unpacks to -1),
// matching the old float image's <0 sky marker that the reproject/accumulate passes early-out on. The
// normal count is clamped to 254 (max packed 1016) so a real count can never collide with the sentinel.
static const uint RTGI_COUNT_NORMAL_MASK = 0x3FFu;  // 10 bits
static const uint RTGI_COUNT_FAST_MASK   = 0x3Fu;   // 6 bits
static const uint RTGI_COUNT_SKY         = 0x3FFu;  // sentinel in the normal field
static const float RTGI_COUNT_NORMAL_MAX = 254.0f;  // highest real normal count; stays below the sentinel

func rtgi_pack_sample_counts(float normal_count, float fast_count) -> uint
{
    const uint n = uint(round(clamp(normal_count, 0.0f, RTGI_COUNT_NORMAL_MAX) * 4.0f)); // 0..1016
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
    return n == RTGI_COUNT_SKY ? -1.0f : float(n) * 0.25f;
}

func rtgi_unpack_fast_count(uint packed) -> float
{
    return float((packed >> 10u) & RTGI_COUNT_FAST_MASK) * 0.25f;
}

struct PixelData
{
    float2 uv;
    float3 ndc;
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
    pd.ndc         = float3(pd.uv * 2.0f - 1.0f, depth_tex[dtid]);
    const float4 pos_pre_div = mul(camera.inv_view_proj, float4(pd.ndc, 1.0f));
    pd.position_ws = pos_pre_div.xyz / pos_pre_div.w;
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
    return inv_exposure * 64.0f;
}

func linear_to_perceptual(float v, float inv_exposure) -> float
{
    return log(max(v, calc_perceptual_radiance_floor(inv_exposure)));
}

__generic<uint N>
func linear_to_perceptual(vector<float, N> v, float inv_exposure) -> vector<float, N>
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

// Max extra rays a disoccluded pixel may request (beyond its 1 base ray). Capped at 31 so a pixel's
// total ray count stays <= 32, matching the trace pass's per-pixel sample_index clamp (min(32u, ...))
// beyond which the decorrelating RNG skip saturates and ray directions would start to duplicate.
#define RTGI_MAX_EXTRA_RAYS 31u

// Fixed uniform ray count per pixel used when ray redistribution is disabled: exactly
// max(floor(ray_budget), 1) rays for every geometry pixel, in both the repacked and classic paths.
func calc_fixed_rays_per_pixel(float ray_percentage) -> uint
{
    return max(uint(floor(max(ray_percentage, 0.0f))), 1u);
}

// Extra rays (beyond the mandatory base ray) a geometry pixel wants, given its reprojected sample count.
// Demand is PROPORTIONAL to how much history the pixel is still missing below fast_convergence_samples
// (its deficit), not a flat "everyone under the target wants the max". A pixel missing 24 requests 24, one
// missing 8 requests 8, so the allocator's demand-proportional budget split hands the scarce rays to the
// freshest disocclusions (e.g. with 16 rays for a 24+8 pair -> 12 and 4) instead of splitting them evenly.
// fast_convergence_samples is the sample count at which demand reaches zero. Capped at RTGI_MAX_EXTRA_RAYS.
func calc_desired_extra_rays(float reproj_sample_count, float fast_convergence_samples) -> uint
{
    const float deficit = fast_convergence_samples - reproj_sample_count;
    return deficit <= 0.0f ? 0u : uint(min(deficit, float(RTGI_MAX_EXTRA_RAYS)));
}

// Total rays a geometry pixel wants this frame: the mandatory base ray plus its adaptive extras. This is
// the shared source of truth for the per-pixel ray demand — the temporal reproject pass (tile demand
// totals) and the allocate pass (per-pixel gs_desired) both call this so the two can never diverge.
func calc_desired_ray_count(float reproj_sample_count, float fast_convergence_samples) -> uint
{
    return 1u + calc_desired_extra_rays(reproj_sample_count, fast_convergence_samples);
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

// Converts a perceptual-space radiance value to a display-ready linear color for debug visualization:
// maps back to linear, applies exposure, and scales by debug_visualization_scale.
func perceptual_radiance_colormap(float perceptual, float exposure) -> float3
{
    return Heatmap(perceptual_to_linear(perceptual) * exposure * 0.0003f);
}