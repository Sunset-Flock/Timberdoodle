#pragma once

#include "rt_lab_blur.inl"

#include "rt_lab_shared.hlsl"
#include "rt_lab_distribution.hlsl"

[[vk::push_constant]] RtLabBlurPush rt_lab_blur_push;

// 5x5 B3 spline kernel weights by |offset| (1/16, 1/4, 3/8, 1/4, 1/16 per axis).
static const float RT_LAB_ATROUS_KERNEL[3] = {3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};

// Weighted sums of a group of taps: color / split by the filter weight (normalized at the end), count by the weight
// relative to the center weight (the effective sample count, not normalized).
struct RtLabSpatialSum
{
    float3 color;
    float2 split;
    float weight;
    float count;
};

func rt_lab_spatial_sum_zero() -> RtLabSpatialSum
{
    return RtLabSpatialSum(float3(0.0f, 0.0f, 0.0f), float2(0.0f, 0.0f), 0.0f, 0.0f);
}

// Sample count budget (fine filter, RtLabSettings::fine_target_samples > 0): taps are accepted ring by ring from the
// center outwards (closest first). A ring that fits is accepted fully; the ring that would cross the target is scaled
// so the pass reaches the target exactly; the rest is dropped. Returns true when the budget is used up.
func rt_lab_accept_ring(inout RtLabSpatialSum total, RtLabSpatialSum ring, float target) -> bool
{
    float accept = 1.0f;
    if (target > 0.0f && ring.count > 0.0f)
    {
        accept = saturate((target - total.count) / ring.count);
    }
    total.color += ring.color * accept;
    total.split += ring.split * accept;
    total.weight += ring.weight * accept;
    total.count += ring.count * accept;
    return target > 0.0f && total.count >= target * 0.9999f;
}

// Variance guide of the fine filter (RtLabSettings::variance_guide_*): the center's mean and the uncertainty of it.
struct RtLabVarianceGuide
{
    bool enabled;
    float3 mean;
    float3 mean_variance; // per sample variance / effective sample count
    float rcp_two_k_sq;   // 1 / (2 k^2)
    float3 variance_floor;
    // Distribution guide (RtLabSettings::distribution_guide_mode): the center's body log mean + log variance. Independent of the variance guide.
    bool dist_enabled;
    float dist_mean;
    float dist_variance;
    float dist_sigmas;    // k
    // Mode 3: the center's three components (dark tail, body, bright tail): weight, log mean, log variance. dist_multi = compare them all.
    bool dist_multi;
    float3 dist_component_weight;
    float3 dist_component_mean;
    float3 dist_component_variance;
    float dist_self_overlap;
    // Mode 1: the comparison (RT_LAB_DISTRIBUTION_COMPARE_*) and the center's mixture, built once per pixel (not per tap).
    int dist_compare;
    RtLabTailMixture dist_mixture;
    // The center's total energy estimate (rt_lab_distribution_energy_of / rt_lab_distribution_energy), a tap has to match it too.
    float dist_log_energy;
    float dist_energy_variance;
};

func rt_lab_variance_guide_none() -> RtLabVarianceGuide
{
    const float3 zero = float3(0.0f, 0.0f, 0.0f);
    return RtLabVarianceGuide(false, zero, zero, 0.0f, zero, false, 0.0f, 0.0f, 1.0f, false, zero, zero, zero, 1.0f, 0, rt_lab_tail_mixture(float4(0.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 0.0f)), 0.0f, 1.0f);
}

// Per sample variance / effective sample count of a pixel (variance guide).
func rt_lab_mean_variance(int2 pixel, float count) -> float3
{
    let push = rt_lab_blur_push;
    return max(push.attach.lab_variance.get()[pixel].rgb, 0.0f) / max(count, 1e-3f);
}

func rt_lab_distribution_guide_weight(RtLabVarianceGuide guide, int2 tap) -> float
{
    let push = rt_lab_blur_push;
    if (guide.dist_multi)
    {
        const float4 body = push.attach.lab_distribution.get()[tap];
        const float4 bright = push.attach.lab_distribution_tail.get()[tap];
        const float4 dark = push.attach.lab_distribution_low.get()[tap];
        if (!(body.w >= 1.0f) || any(!isfinite(body)) || any(!isfinite(bright)) || any(!isfinite(dark)))
        {
            return 1.0f;
        }
        // The energy test first (cheap, rejects most taps across an edge): the overlap sees the components by their probability mass, the tails are rare and
        // carry the energy, the tap also has to match the center's total energy.
        RtLabEnergy center_energy;
        center_energy.log_energy = guide.dist_log_energy;
        center_energy.relative_variance = guide.dist_energy_variance;
        const float energy_weight = rt_lab_distribution_energy_weight(center_energy, rt_lab_distribution_energy(body, bright, dark), guide.dist_sigmas);
        if (energy_weight < RT_LAB_DISTRIBUTION_EARLY_OUT)
        {
            return energy_weight;
        }
        float3 tap_weight, tap_mean, tap_variance;
        rt_lab_distribution_components(body, bright, dark, tap_weight, tap_mean, tap_variance);
        const float cross = rt_lab_distribution_overlap(guide.dist_component_weight, guide.dist_component_mean, guide.dist_component_variance, tap_weight, tap_mean, tap_variance, guide.dist_sigmas);
        const float self = rt_lab_distribution_self_overlap(tap_weight, tap_mean, tap_variance, guide.dist_sigmas);
        // Cosine-like normalization: 1 for identical mixtures, ~0 for mixtures without a matching lump.
        return cross * rsqrt(max(guide.dist_self_overlap * self, 1e-8f)) * energy_weight;
    }
    const float4 state = push.attach.lab_distribution.get()[tap];
    if (!(state.w >= 1.0f) || any(!isfinite(state)))
    {
        return 1.0f;
    }
    RtLabEnergy center_energy;
    center_energy.log_energy = guide.dist_log_energy;
    center_energy.relative_variance = guide.dist_energy_variance;
    return rt_lab_distribution_compare_cached(guide.dist_compare, guide.dist_mixture, center_energy, state, push.attach.lab_distribution_tail.get()[tap], guide.dist_sigmas);
}

// The filtered color of a pass: the body part's weighted mean + the tail part's weighted mean (no tail split: the plain mean).
func rt_lab_spatial_filtered_color(RtLabSpatialSum total) -> float3
{
    return total.weight > 0.0f ? total.color / total.weight : float3(0.0f, 0.0f, 0.0f);
}

func rt_lab_variance_guide_begin(uint2 dtid, float3 mean, float count) -> RtLabVarianceGuide
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    RtLabVarianceGuide guide = rt_lab_variance_guide_none();
    if (count <= 0.0f)
    {
        return guide;
    }
    if (push.variance_bound != 0u && settings.variance_guide_enabled != 0)
    {
        guide.enabled = true;
        guide.mean = mean;
        guide.mean_variance = rt_lab_mean_variance(int2(dtid), count);
        const float k = max(settings.variance_guide_sigmas, 1e-3f);
        guide.rcp_two_k_sq = rcp(2.0f * k * k);
        guide.variance_floor = rt_lab_variance_guide_floor(push.attach.globals.exposure);
    }
    if (push.distribution_bound != 0u && settings.distribution_guide_mode != 0)
    {
        const float4 state = push.attach.lab_distribution.get()[dtid];
        if (state.w >= 1.0f && all(isfinite(state)))
        {
            guide.dist_enabled = true;
            guide.dist_mean = state.x;
            guide.dist_variance = rt_lab_distribution_variance(state);
            guide.dist_sigmas = max(settings.distribution_guide_sigmas, 1e-3f);
            if (push.distribution_multi == 0u)
            {
                guide.dist_compare = settings.distribution_guide_compare;
                guide.dist_mixture = rt_lab_tail_mixture(state, push.attach.lab_distribution_tail.get()[dtid]);
                const RtLabEnergy center_energy = rt_lab_distribution_energy_of(guide.dist_mixture);
                guide.dist_log_energy = center_energy.log_energy;
                guide.dist_energy_variance = center_energy.relative_variance;
            }
            if (push.distribution_multi != 0u)
            {
                guide.dist_multi = true;
                rt_lab_distribution_components(state, push.attach.lab_distribution_tail.get()[dtid], push.attach.lab_distribution_low.get()[dtid],
                    guide.dist_component_weight, guide.dist_component_mean, guide.dist_component_variance);
                guide.dist_self_overlap = rt_lab_distribution_self_overlap(guide.dist_component_weight, guide.dist_component_mean, guide.dist_component_variance, guide.dist_sigmas);
                const RtLabEnergy energy = rt_lab_distribution_energy(state, push.attach.lab_distribution_tail.get()[dtid], push.attach.lab_distribution_low.get()[dtid]);
                guide.dist_log_energy = energy.log_energy;
                guide.dist_energy_variance = energy.relative_variance;
            }
        }
    }
    return guide;
}

// Base shapes of the SH guide modes (the weight is pow(base, the mode's strength)).
#define RT_LAB_SH_GUIDE_DIRECTION_EPSILON 0.05f // direction difference (unit vectors) that always passes the directional variance test
#define RT_LAB_SH_GUIDE_CROSS_TOLERANCE 0.5f      // relative luminance difference at which the base weight is exp(-1)
#define RT_LAB_SH_GUIDE_STATISTICAL_SIGMAS 2.0f

// SH guide (RtLabSettings::sh_guide_mode): where the light of a pixel comes from, from the temporal pass's SH-Y (xyz =
// direction x Y, w = Y; the unfiltered temporal accumulate, read only) and CoCg (.z = history length).
struct RtLabShGuide
{
    bool enabled;
    float4 sh_y;
    float history;       // history length (rays)
    float3 normal;       // center face normal
    float concentration; // debiased R of the center
    float resolved_luma; // center SH resolved at its own normal
    RtLabShMoments moments; // direction covariance + luma variance (RT_LAB_SH_GUIDE_DIRECTIONAL_VARIANCE)
};

func rt_lab_sh_guide_none() -> RtLabShGuide
{
    RtLabShGuide guide = {};
    guide.normal = float3(0.0f, 0.0f, 1.0f);
    return guide;
}

func rt_lab_sh_guide_begin(uint2 dtid, float3 normal) -> RtLabShGuide
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (push.sh_bound == 0u || (settings.sh_guide_mode & RT_LAB_SH_GUIDE_MODE_MASK) == RT_LAB_SH_GUIDE_OFF)
    {
        return rt_lab_sh_guide_none();
    }
    const float4 sh_y = push.attach.lab_sh_y.get()[dtid];
    const float4 cocg = push.attach.lab_cocg.get()[dtid];
    if (!(sh_y.w > 0.0f))
    {
        return rt_lab_sh_guide_none(); // no rays yet: no opinion
    }
    RtLabShGuide guide;
    guide.enabled = true;
    guide.sh_y = sh_y;
    guide.history = cocg.z;
    guide.normal = normal;
    guide.concentration = rt_lab_sh_concentration(sh_y, cocg.z);
    guide.resolved_luma = rt_lab_luminance(rt_lab_sh_resolve(sh_y, cocg.xy, normal));
    if (push.sh_m2_bound != 0u)
    {
        guide.moments = rt_lab_sh_moments_decode(sh_y, push.attach.lab_sh_m2.get()[dtid], push.attach.lab_sh_m2b.get()[dtid]);
    }
    return guide;
}

// Decoded second moments of one pixel's light direction distribution (RT_LAB_SH_GUIDE_DIRECTIONAL_VARIANCE): the Y weighted
// covariance of the direction C = E[Y d d^T] / E[Y] - m m^T, m = xyz / w, as diagonal (xx, yy, zz) and off diagonal (xy, xz, yz),
// and the luma variance per sample E[Y^2] - E[Y]^2.
struct RtLabShMoments
{
    float3 diagonal;
    float3 off_diagonal;
    float luma_variance;
};

func rt_lab_sh_moments_decode(float4 sh_y, float4 m2_a, float4 m2_b) -> RtLabShMoments
{
    const float rcp_y = rcp(max(sh_y.w, 1e-12f));
    const float3 mean = sh_y.xyz * rcp_y;
    RtLabShMoments moments;
    moments.diagonal = max(float3(m2_a.x, m2_a.y, m2_a.z) * rcp_y - mean * mean, 0.0f);
    moments.off_diagonal = float3(m2_a.w, m2_b.x, m2_b.y) * rcp_y - float3(mean.x * mean.y, mean.x * mean.z, mean.y * mean.z);
    moments.luma_variance = max(m2_b.z - sh_y.w * sh_y.w, 0.0f);
    return moments;
}

// d^T S^-1 d for the symmetric 3x3 S = [[a, b, c], [b, d, e], [c, e, f]] (adjugate form, S is positive definite).
func rt_lab_quadratic_inverse(float3 diagonal, float3 off_diagonal, float3 v) -> float
{
    const float a = diagonal.x;
    const float d = diagonal.y;
    const float f = diagonal.z;
    const float b = off_diagonal.x;
    const float c = off_diagonal.y;
    const float e = off_diagonal.z;
    const float i00 = d * f - e * e;
    const float i01 = c * e - b * f;
    const float i02 = b * e - c * d;
    const float i11 = a * f - c * c;
    const float i12 = b * c - a * e;
    const float i22 = a * d - b * b;
    const float determinant = a * i00 + b * i01 + c * i02;
    const float quadratic = v.x * (i00 * v.x + i01 * v.y + i02 * v.z) + v.y * (i01 * v.x + i11 * v.y + i12 * v.z) + v.z * (i02 * v.x + i12 * v.y + i22 * v.z);
    return quadratic / max(determinant, 1e-20f);
}

// Tap weight factor of the SH guide (1 = no opinion).
func rt_lab_sh_guide_weight(RtLabShGuide guide, int2 tap) -> float
{
    if (!guide.enabled)
    {
        return 1.0f;
    }
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    const float4 tap_sh_y = push.attach.lab_sh_y.get()[tap];
    if (!(tap_sh_y.w > 0.0f))
    {
        return 1.0f;
    }
    const float4 tap_cocg = push.attach.lab_cocg.get()[tap];
    // Every mode computes a base weight in 0..1 and returns pow(base, its strength): strength 0 = no effect, higher = stronger.
    const int sh_mode = settings.sh_guide_mode & RT_LAB_SH_GUIDE_MODE_MASK;
    float strength = settings.sh_guide_statistical_strength;
    if (sh_mode == RT_LAB_SH_GUIDE_DIRECTION)
    {
        strength = settings.sh_guide_direction_strength;
    }
    else if (sh_mode == RT_LAB_SH_GUIDE_CROSS)
    {
        strength = settings.sh_guide_cross_strength;
    }
    else if (sh_mode == RT_LAB_SH_GUIDE_DIRECTIONAL_VARIANCE)
    {
        strength = settings.sh_guide_directional_variance_strength;
    }
    strength = max(strength, 0.0f);
    if (strength <= 0.0f)
    {
        return 1.0f; // pow(0, 0) is undefined
    }
    if (sh_mode == RT_LAB_SH_GUIDE_DIRECTION)
    {
        // Dominant direction agreement, only as far as BOTH pixels have a clear dominant direction.
        const float confidence = min(guide.concentration, rt_lab_sh_concentration(tap_sh_y, tap_cocg.z));
        const float3 center_dir = guide.sh_y.xyz * rsqrt(max(dot(guide.sh_y.xyz, guide.sh_y.xyz), 1e-12f));
        const float3 tap_dir = tap_sh_y.xyz * rsqrt(max(dot(tap_sh_y.xyz, tap_sh_y.xyz), 1e-12f));
        return pow(lerp(1.0f, saturate(dot(center_dir, tap_dir)), confidence), strength);
    }
    if (sh_mode == RT_LAB_SH_GUIDE_CROSS)
    {
        // The tap's light as seen from the center's surface vs the center's own.
        const float tap_luma = rt_lab_luminance(rt_lab_sh_resolve(tap_sh_y, tap_cocg.xy, guide.normal));
        const float luma_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
        const float relative = abs(guide.resolved_luma - tap_luma) / (max(guide.resolved_luma, tap_luma) + luma_floor);
        return pow(exp(-square(relative / RT_LAB_SH_GUIDE_CROSS_TOLERANCE)), strength);
    }
    if (sh_mode == RT_LAB_SH_GUIDE_DIRECTIONAL_VARIANCE)
    {
        // Mean light direction m = xyz / w and mean Y of both pixels vs their own noise. Noise of a mean = the per sample moment
        // / history length: the direction covariance C (anisotropic) -> Mahalanobis distance of the direction difference under
        // C_c / n_c + C_t / n_t + eps^2 I (eps = how much direction difference always passes), and the luma variance -> z test of Y
        // (floor keeps dark pixels finite). z^2 per degree of freedom (3 + 1).
        if (push.sh_m2_bound == 0u)
        {
            return 1.0f; // no second moments (graph without them)
        }
        const float luma_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
        const RtLabShMoments tap_moments = rt_lab_sh_moments_decode(tap_sh_y, push.attach.lab_sh_m2.get()[tap], push.attach.lab_sh_m2b.get()[tap]);
        const float rcp_center_n = rcp(max(guide.history, 1.0f));
        const float rcp_tap_n = rcp(max(tap_cocg.z, 1.0f));
        const float3 direction_difference = guide.sh_y.xyz / guide.sh_y.w - tap_sh_y.xyz / tap_sh_y.w;
        const float3 diagonal = guide.moments.diagonal * rcp_center_n + tap_moments.diagonal * rcp_tap_n + square(RT_LAB_SH_GUIDE_DIRECTION_EPSILON);
        const float3 off_diagonal = guide.moments.off_diagonal * rcp_center_n + tap_moments.off_diagonal * rcp_tap_n;
        const float z_direction_sq = rt_lab_quadratic_inverse(diagonal, off_diagonal, direction_difference);
        const float z_luma_sq = square(guide.sh_y.w - tap_sh_y.w) / (guide.moments.luma_variance * rcp_center_n + tap_moments.luma_variance * rcp_tap_n + square(luma_floor));
        return pow(exp(-(z_direction_sq + z_luma_sq) * 0.25f / 8.0f), strength);
    }
    // STATISTICAL: z test of the mean light vectors; per ray direction noise of unit vectors = 1 - R^2.
    const float3 center_mean = guide.sh_y.xyz / guide.sh_y.w;
    const float3 tap_mean = tap_sh_y.xyz / tap_sh_y.w;
    const float direction_variance = max(1.0f - 0.5f * (dot(center_mean, center_mean) + dot(tap_mean, tap_mean)), 0.05f);
    const float z_sq = dot(center_mean - tap_mean, center_mean - tap_mean) /
                       (direction_variance * (1.0f / max(guide.history, 1.0f) + 1.0f / max(tap_cocg.z, 1.0f)));
    return pow(exp(-z_sq / (2.0f * RT_LAB_SH_GUIDE_STATISTICAL_SIGMAS * RT_LAB_SH_GUIDE_STATISTICAL_SIGMAS)), strength);
}

// One spatial tap: filter weight (kernel x edge stopping (plane distance + normal power), 0 = rejected) and its color /
// count / split added to ring.
func rt_lab_spatial_tap(inout RtLabSpatialSum ring, int2 tap, float kernel, float count_scale,
    float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, bool clamp_count, RtLabVarianceGuide guide, RtLabShGuide sh_guide)
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (any(tap < 0) || any(tap >= int2(push.size)))
    {
        return;
    }
    const float tap_depth = push.attach.lab_depth.get()[tap];
    if (tap_depth == 0.0f)
    {
        return;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
    const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
    float weight = kernel *
                   rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, settings.blur_plane_tolerance) *
                   pow(saturate(dot(normal, tap_normal)), settings.blur_normal_power);
    if (weight <= 1e-6f)
    {
        return;
    }
    const float4 value = push.attach.src.get()[tap];
    const float tap_count = rt_lab_spatial_read_count(value.a, clamp_count, settings.sample_count_input_clamp);
    if (tap_count <= 0.0f)
    {
        return; // no samples (ray redistribution left the pixel without a ray)
    }
    // A rejected tap is not gathered: the guide weights also scale the effective sample count.
    if (sh_guide.enabled)
    {
        weight *= rt_lab_sh_guide_weight(sh_guide, tap);
        if (weight <= 1e-6f)
        {
            return;
        }
    }
    if (guide.enabled)
    {
        // Filtering the raw frame: the tap's guide mean + history length are the accumulated statistics' (a tap without history has no opinion).
        float3 guide_value = value.rgb;
        float guide_count = tap_count;
        if (push.guide_mean_bound != 0u)
        {
            const float4 tap_guide = push.attach.lab_guide_mean.get()[tap];
            guide_value = tap_guide.rgb;
            guide_count = tap_guide.a;
        }
        if (guide_count > 0.0f)
        {
            weight *= rt_lab_variance_guide_likelihood(guide.mean, guide.mean_variance, guide_value, rt_lab_mean_variance(tap, guide_count),
                guide.variance_floor, guide.rcp_two_k_sq);
            if (weight <= 1e-6f)
            {
                return;
            }
        }
    }
    // Distribution guide (rt_lab_distribution.hlsl): the tap's tracked state against the center's.
    if (guide.dist_enabled)
    {
        weight *= rt_lab_distribution_guide_weight(guide, tap);
        if (weight <= 1e-6f)
        {
            return;
        }
    }
    ring.color += value.rgb * weight;
    ring.weight += weight;
    ring.count += tap_count * weight * count_scale;
    if (push.split_enabled != 0u)
    {
        ring.split += push.attach.split_src.get()[tap] * weight;
    }
}

// Shared prologue of both filters: false = sky (output written).
func rt_lab_spatial_begin(uint2 dtid, out float4 center, out float3 position, out float3 normal, out float rcp_pixel_width_ws, out float2 inv_size) -> bool
{
    let push = rt_lab_blur_push;
    center = push.attach.src.get()[dtid];
    position = float3(0.0f, 0.0f, 0.0f);
    normal = float3(0.0f, 0.0f, 1.0f);
    rcp_pixel_width_ws = 0.0f;
    inv_size = rcp(float2(push.size));
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (push.split_enabled != 0u)
        {
            push.attach.split_dst.get()[dtid] = float2(0.0f, 0.0f);
        }
        return false;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.size.y), depth));
    return true;
}

// Writes the pass output.
func rt_lab_spatial_end(uint2 dtid, RtLabSpatialSum total)
{
    let push = rt_lab_blur_push;
    // No samples anywhere in the footprint: count 0, the next pass / the resolve skip it.
    const float rcp_weight = total.weight > 0.0f ? rcp(total.weight) : 0.0f;
    push.attach.dst.get()[dtid] = float4(total.weight > 0.0f ? rt_lab_spatial_filtered_color(total) : float3(0.0f, 0.0f, 0.0f), total.count / RT_LAB_SPATIAL_COUNT_SCALE);
    if (push.split_enabled != 0u)
    {
        push.attach.split_dst.get()[dtid] = total.split * rcp_weight;
    }
}

// The center tap: always fully accepted (relative count weight 1), unless it holds no samples (weight 0).
func rt_lab_spatial_center(uint2 dtid, float4 center, float weight_in, bool clamp_count) -> RtLabSpatialSum
{
    let push = rt_lab_blur_push;
    RtLabSpatialSum total = rt_lab_spatial_sum_zero();
    total.count = rt_lab_spatial_read_count(center.a, clamp_count, push.attach.globals.rt_lab_settings.sample_count_input_clamp);
    const float weight = total.count > 0.0f ? weight_in : 0.0f;
    total.color = center.rgb * weight;
    total.weight = weight;
    if (push.split_enabled != 0u)
    {
        total.split = push.attach.split_src.get()[dtid] * weight;
    }
    return total;
}

// Early out (RtLabSettings::fine_target_early_out): the center already holds the target sample count, the pass would
// return it anyway. Writes the pass output and returns true.
func rt_lab_spatial_early_out(uint2 dtid, float4 center, bool clamp_count) -> bool
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (settings.fine_target_early_out == 0 || settings.fine_target_samples <= 0.0f)
    {
        return false;
    }
    const RtLabSpatialSum total = rt_lab_spatial_center(dtid, center, 1.0f, clamp_count);
    if (total.count < settings.fine_target_samples)
    {
        return false;
    }
    rt_lab_spatial_end(dtid, total);
    return true;
}

// Variance guide setup shared by both filters.
func rt_lab_spatial_guide(uint2 dtid, float4 center, bool clamp_count) -> RtLabVarianceGuide
{
    let push = rt_lab_blur_push;
    const float center_count = rt_lab_spatial_read_count(center.a, clamp_count, push.attach.globals.rt_lab_settings.sample_count_input_clamp);
    if (push.guide_mean_bound != 0u)
    {
        // Filtering the raw frame: the guide mean + history length are the accumulated statistics' of this pixel.
        const float4 guide_mean = push.attach.lab_guide_mean.get()[dtid];
        return rt_lab_variance_guide_begin(dtid, guide_mean.rgb, center_count > 0.0f ? guide_mean.a : 0.0f);
    }
    return rt_lab_variance_guide_begin(dtid, center.rgb, center_count);
}

// One a-trous pass: 5x5 B3 spline taps spaced push.step pixels apart, rings (max(|x|, |y|) = 1, then 2) accepted from
// the center outwards under the fine filter's sample count budget. The split difference takes the same weights.
// Alpha: effective sample count.
[shader("compute")]
[numthreads(RT_LAB_BLUR_X, RT_LAB_BLUR_Y, 1)]
func entry_rt_lab_atrous(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    float4 center;
    float3 position;
    float3 normal;
    float rcp_pixel_width_ws;
    float2 inv_size;
    if (!rt_lab_spatial_begin(dtid, center, position, normal, rcp_pixel_width_ws, inv_size))
    {
        return;
    }
    const bool clamp_count = push.clamp_input_count != 0u;
    if (rt_lab_spatial_early_out(dtid, center, clamp_count))
    {
        return;
    }
    const float target = push.attach.globals.rt_lab_settings.fine_target_samples;
    const float center_kernel = RT_LAB_ATROUS_KERNEL[0] * RT_LAB_ATROUS_KERNEL[0];
    const RtLabVarianceGuide guide = rt_lab_spatial_guide(dtid, center, clamp_count);
    RtLabSpatialSum total = rt_lab_spatial_center(dtid, center, center_kernel, clamp_count);
    const RtLabShGuide sh_guide = rt_lab_sh_guide_begin(dtid, normal);
    if (!(target > 0.0f && total.count >= target))
    {
        [unroll] for (int r = 1; r <= 2; ++r)
        {
            RtLabSpatialSum ring = rt_lab_spatial_sum_zero();
            [unroll] for (int y = -2; y <= 2; ++y)
            {
                [unroll] for (int x = -2; x <= 2; ++x)
                {
                    if (max(abs(x), abs(y)) != r)
                    {
                        continue;
                    }
                    const float kernel = RT_LAB_ATROUS_KERNEL[abs(x)] * RT_LAB_ATROUS_KERNEL[abs(y)];
                    rt_lab_spatial_tap(ring, int2(dtid) + int2(x, y) * push.step, kernel, 1.0f / center_kernel,
                        position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide, sh_guide);
                }
            }
            if (rt_lab_accept_ring(total, ring, target))
            {
                break;
            }
        }
    }
    rt_lab_spatial_end(dtid, total);
}

// One pass of the separable bilateral gaussian. Edge stopping is purely geometric: plane distance (in pixel widths)
// and face normal similarity, plus the variance guide. Taps +-d are accepted from d = 1 outwards under the fine
// filter's sample count budget. Alpha: effective sample count (rt_lab_blur.inl).
[shader("compute")]
[numthreads(RT_LAB_BLUR_X, RT_LAB_BLUR_Y, 1)]
func entry_rt_lab_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    float4 center;
    float3 position;
    float3 normal;
    float rcp_pixel_width_ws;
    float2 inv_size;
    if (!rt_lab_spatial_begin(dtid, center, position, normal, rcp_pixel_width_ws, inv_size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const int radius = clamp(settings.blur_radius, 0, 32);
    const float sigma = max(float(radius) * 0.5f, 0.5f);
    const float rcp_two_sigma_sq = rcp(2.0f * sigma * sigma);
    const bool clamp_count = push.clamp_input_count != 0u;
    if (rt_lab_spatial_early_out(dtid, center, clamp_count))
    {
        return;
    }
    const float target = settings.fine_target_samples;

    // The center weight is 1, so the relative count weight is the weight itself.
    const RtLabVarianceGuide guide = rt_lab_spatial_guide(dtid, center, clamp_count);
    RtLabSpatialSum total = rt_lab_spatial_center(dtid, center, 1.0f, clamp_count);
    const RtLabShGuide sh_guide = rt_lab_sh_guide_begin(dtid, normal);
    if (!(target > 0.0f && total.count >= target))
    {
        [loop] for (int d = 1; d <= radius; ++d)
        {
            const float kernel = exp(-float(d * d) * rcp_two_sigma_sq);
            RtLabSpatialSum ring = rt_lab_spatial_sum_zero();
            rt_lab_spatial_tap(ring, int2(dtid) - push.direction * d, kernel, 1.0f, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide, sh_guide);
            rt_lab_spatial_tap(ring, int2(dtid) + push.direction * d, kernel, 1.0f, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide, sh_guide);
            if (rt_lab_accept_ring(total, ring, target))
            {
                break;
            }
        }
    }
    rt_lab_spatial_end(dtid, total);
}
