#pragma once

#include "rt_lab_blur.inl"

#include "rt_lab_shared.hlsl"

// Distribution guide comparisons (RtLabSettings::distribution_guide_mode, rt_lab.md "Distribution guide"). Pure functions of the tracked
// per pixel states, shared by the spatial filter (rt_lab_blur.hlsl) and the live pixel probes (rt_lab_probe.hlsl).
//
// Pixel state (rt_lab_temporal.inl), all in y = ln(1 + luma / floor), so e^y = 1 + luma / floor ("floor units"):
//   A = (body mean, body variance, perceptual mean of all samples, body sample count)
//   B = (tail fraction p, tail mean, tail variance, tail sample count)      (tail count < 1 = no tail)
//   C = (dark tail fraction, mean, variance, count)                         (mode 3 only)

// Per sample log variance of a distribution state (A: mean, variance, perceptual mean, count) with one pseudo sample of the prior
// RT_LAB_DISTRIBUTION_PRIOR_VARIANCE: a body of a single sample has variance 0, which would reject every neighbor.
func rt_lab_distribution_variance(float4 state) -> float
{
    return (state.w * max(state.y, 0.0f) + RT_LAB_DISTRIBUTION_PRIOR_VARIANCE) / (state.w + 1.0f);
}

// The three components of a pixel (dark tail, body, bright tail): fractions (the body gets what the tails leave), log means, log variances with the prior.
func rt_lab_distribution_components(float4 body, float4 bright, float4 dark, out float3 weight, out float3 mean, out float3 variance)
{
    const float bright_fraction = bright.w >= 1.0f ? saturate(bright.x) : 0.0f;
    const float dark_fraction = dark.w >= 1.0f ? saturate(dark.x) : 0.0f;
    weight = float3(dark_fraction, max(1.0f - bright_fraction - dark_fraction, 0.0f), bright_fraction);
    mean = float3(dark.y, body.x, bright.y);
    variance = float3(rt_lab_distribution_variance(float4(0.0f, dark.z, 0.0f, dark.w)), rt_lab_distribution_variance(body), rt_lab_distribution_variance(float4(0.0f, bright.z, 0.0f, bright.w)));
}

// Probability-like overlap of two pixels' mixtures: sum over component pairs of w_a w_b exp(-|mu_a - mu_b| / (k sqrt(var_a + var_b) + eps)).
func rt_lab_distribution_overlap(float3 weight_a, float3 mean_a, float3 variance_a, float3 weight_b, float3 mean_b, float3 variance_b, float k) -> float
{
    float sum = 0.0f;
    [unroll] for (int i = 0; i < 3; ++i)
    {
        [unroll] for (int j = 0; j < 3; ++j)
        {
            if (weight_a[i] * weight_b[j] <= 0.0f)
            {
                continue; // an empty component (no dark / bright tail): no exp
            }
            sum += weight_a[i] * weight_b[j] * exp(-abs(mean_a[i] - mean_b[j]) / (k * sqrt(variance_a[i] + variance_b[j]) + RT_LAB_DISTRIBUTION_EPSILON));
        }
    }
    return sum;
}

// The overlap of a mixture with itself: the diagonal terms are exp(0) = 1, the off diagonal pairs are symmetric (3 exps instead of the 9 of the general form).
func rt_lab_distribution_self_overlap(float3 weight, float3 mean, float3 variance, float k) -> float
{
    float sum = weight.x * weight.x + weight.y * weight.y + weight.z * weight.z;
    [unroll] for (int i = 0; i < 3; ++i)
    {
        [unroll] for (int j = i + 1; j < 3; ++j)
        {
            if (weight[i] * weight[j] <= 0.0f)
            {
                continue;
            }
            sum += 2.0f * weight[i] * weight[j] * exp(-abs(mean[i] - mean[j]) / (k * sqrt(variance[i] + variance[j]) + RT_LAB_DISTRIBUTION_EPSILON));
        }
    }
    return sum;
}

// == Lognormal body + bright tail (mode 1) ======================================================================================================

// A pixel as the two component mixture (1 - p) body + p tail.
struct RtLabTailMixture
{
    float p;            // tail fraction (0 = no tail)
    float2 mean;        // log means (body, tail)
    float2 variance;    // log variances with the prior (body, tail)
    float2 first;       // E[e^y] of each component = exp(mu + var / 2): its mean in floor units (1 + luma / floor)
    float2 energy;      // fraction x first: the component's share of the mixture mean
    float2 samples;     // samples behind each component's estimate (body count, tail count)
    float perceptual;   // the tracked perceptual mean of all samples (state A.z)
    float count;        // samples behind the mixture (body + tail count)
    float mean_energy;  // total energy = E[e^y] of the mixture = 1 + mean luma / floor
};

func rt_lab_tail_mixture(float4 body, float4 tail) -> RtLabTailMixture
{
    RtLabTailMixture m = {};
    const bool has_tail = tail.w >= 1.0f && all(isfinite(tail));
    m.p = has_tail ? saturate(tail.x) : 0.0f;
    m.mean = float2(body.x, has_tail ? tail.y : 0.0f);
    m.variance = float2(rt_lab_distribution_variance(body), rt_lab_distribution_variance(float4(0.0f, tail.z, 0.0f, tail.w)));
    // exp arguments are capped: a tail mean of 20 in y is already a luminance of 5e8 floors.
    m.first = float2(exp(min(m.mean.x + 0.5f * m.variance.x, 20.0f)), has_tail ? exp(min(m.mean.y + 0.5f * m.variance.y, 20.0f)) : 0.0f);
    m.energy = float2((1.0f - m.p) * m.first.x, m.p * m.first.y);
    m.samples = float2(max(body.w, 1.0f), has_tail ? tail.w : 0.0f);
    m.perceptual = body.z;
    m.count = max(body.w + m.samples.y, 1.0f);
    m.mean_energy = m.energy.x + m.energy.y;
    return m;
}

// The two sided Welch style test of the body log means only, exp(-|mu_c - mu_t| / (k sqrt(var_c + var_t) + eps)). The tail is ignored: a pixel
// whose rays are 90 % the same as its neighbor's but is in shadow of the bright light passes.
func rt_lab_distribution_compare_body(RtLabTailMixture c, RtLabTailMixture t, float k) -> float
{
    return exp(-abs(c.mean.x - t.mean.x) / (k * sqrt(c.variance.x + t.variance.x) + RT_LAB_DISTRIBUTION_EPSILON));
}

// Total log variance of the mixture (law of total variance), the spread of ALL the pixel's samples in y.
func rt_lab_distribution_total_log_variance(RtLabTailMixture m) -> float
{
    const float mean = (1.0f - m.p) * m.mean.x + m.p * m.mean.y;
    return (1.0f - m.p) * (m.variance.x + square(m.mean.x - mean)) + m.p * (m.variance.y + square(m.mean.y - mean));
}

// The donors the weights test are whole distributions, not just their bodies: a donor whose body matches but whose rare bright rays come from a light the
// receiver does not see would put those rays into the receiver.

// Unnormalized membership of the log value x in a pixel's mixture: sum over components of (fraction) exp(-z^2 / 2), z in units of k standard deviations.
func rt_lab_distribution_membership(RtLabTailMixture m, float x, float k) -> float
{
    const float2 z = (float2(x, x) - m.mean) / (k * sqrt(m.variance));
    return (1.0f - m.p) * exp(-0.5f * z.x * z.x) + m.p * exp(-0.5f * z.y * z.y);
}

// How well the donor's typical values (body AND tail, by their mass) are explained by the receiver's mixture.
func rt_lab_distribution_membership_of(RtLabTailMixture donor, RtLabTailMixture receiver, float k) -> float
{
    return (1.0f - donor.p) * rt_lab_distribution_membership(receiver, donor.mean.x, k) + donor.p * rt_lab_distribution_membership(receiver, donor.mean.y, k);
}

// 1 MEMBERSHIP: sqrt(donor in receiver x receiver in donor) divided by the same for each pixel in itself, so identical pixels give 1. A donor tail the
// receiver cannot explain costs that tail's mass (and the other way around), a tail both have costs nothing. Symmetric.
func rt_lab_distribution_compare_membership(RtLabTailMixture c, RtLabTailMixture t, float k) -> float
{
    const float cross = sqrt(rt_lab_distribution_membership_of(t, c, k) * rt_lab_distribution_membership_of(c, t, k));
    const float self = sqrt(rt_lab_distribution_membership_of(c, c, k) * rt_lab_distribution_membership_of(t, t, k));
    return saturate(cross / max(self, 1e-6f));
}

// The Welch style test on the tracked perceptual means (the mean of y over ALL samples, no body / tail split involved), against the total
// log spread of both pixels (body AND tail). Nothing in this test depends on the classification (a tail that is not found yet cannot make it wrong), but it is
// a test of the mean only (see rt_lab_distribution_compare_perceptual_mean for the version that also tests body and tail).
func rt_lab_distribution_perceptual_mean_weight(RtLabTailMixture c, RtLabTailMixture t, float k) -> float
{
    return exp(-abs(c.perceptual - t.perceptual) / (k * sqrt(rt_lab_distribution_total_log_variance(c) + rt_lab_distribution_total_log_variance(t)) + RT_LAB_DISTRIBUTION_EPSILON));
}

// 3 BODY + TAIL: the body Welch test and a tail test blended by the share s of the mean energy the tails carry (average of both pixels):
// (1 - s) body + s tail. The tail test is the Welch test of the tail log means when both have a tail; when only one has one, the other lacks that
// tail's share of the energy: 1 - share. A tail that carries 70 % of the light decides 70 % of the weight.
func rt_lab_distribution_compare_body_tail(RtLabTailMixture c, RtLabTailMixture t, float k) -> float
{
    const float w_body = rt_lab_distribution_compare_body(c, t, k);
    const float share_c = c.energy.y / max(c.mean_energy, 1e-6f);
    const float share_t = t.energy.y / max(t.mean_energy, 1e-6f);
    float w_tail = 1.0f;
    if (c.samples.y >= 1.0f && t.samples.y >= 1.0f)
    {
        w_tail = exp(-abs(c.mean.y - t.mean.y) / (k * sqrt(c.variance.y + t.variance.y) + RT_LAB_DISTRIBUTION_EPSILON));
    }
    else if (c.samples.y >= 1.0f)
    {
        w_tail = 1.0f - share_c;
    }
    else if (t.samples.y >= 1.0f)
    {
        w_tail = 1.0f - share_t;
    }
    return lerp(w_body, w_tail, 0.5f * (share_c + share_t));
}

// 2 PERCEPTUAL MEAN: the geometric mean of the perceptual mean test (the overall position of all samples, no classification involved) and the body + tail
// test (body AND tail means, blended by the tails' energy share): the donor must match the receiver's overall level AND each of its components.
func rt_lab_distribution_compare_perceptual_mean(RtLabTailMixture c, RtLabTailMixture t, float k) -> float
{
    return sqrt(rt_lab_distribution_perceptual_mean_weight(c, t, k) * rt_lab_distribution_compare_body_tail(c, t, k));
}

// == Total energy ===================================================================================================================================
// The energy a pixel's tracked distribution stands for, E = the mean of exp(y) = 1 + luma / floor over ALL its components (body, bright tail, dark tail), each
// component's mean exp(mu + var / 2) (lognormal). The tail is rare, so a comparison of the probability MASS (the CDFs, the component overlaps) barely sees it, but
// it carries most of the energy: a pixel in dark shadow and one in the light can have the same body and tails that differ by a factor 10 in energy. This is the
// test that has to catch it. It compares the energies of the DISTRIBUTIONS, not the pixels' current colors.
struct RtLabEnergy
{
    float log_energy;        // ln E
    float relative_variance; // Var(E) / E^2, the uncertainty of the estimate (delta method, from the sample counts behind every component and fraction)
};

// bright / dark = states B / C (tail count < 1 = no such tail). body = state A.
func rt_lab_distribution_energy(float4 body, float4 bright, float4 dark) -> RtLabEnergy
{
    const bool has_bright = bright.w >= 1.0f && all(isfinite(bright));
    const bool has_dark = dark.w >= 1.0f && all(isfinite(dark));
    const float p_bright = has_bright ? saturate(bright.x) : 0.0f;
    const float p_dark = has_dark ? saturate(dark.x) : 0.0f;
    const float w_body = max(1.0f - p_bright - p_dark, 0.0f);
    const float variance_body = rt_lab_distribution_variance(body);
    const float variance_bright = rt_lab_distribution_variance(float4(0.0f, bright.z, 0.0f, bright.w));
    const float variance_dark = rt_lab_distribution_variance(float4(0.0f, dark.z, 0.0f, dark.w));
    // exp arguments are capped: a mean of 20 in y is already a luminance of 5e8 floors.
    const float first_body = exp(min(body.x + 0.5f * variance_body, 20.0f));
    const float first_bright = has_bright ? exp(min(bright.y + 0.5f * variance_bright, 20.0f)) : 0.0f;
    const float first_dark = has_dark ? exp(min(dark.y + 0.5f * variance_dark, 20.0f)) : 0.0f;
    const float energy_body = w_body * first_body;
    const float energy_bright = p_bright * first_bright;
    const float energy_dark = p_dark * first_dark;
    const float energy = max(energy_body + energy_bright + energy_dark, 1e-6f);
    // Var(ln mean of a lognormal) = (var + var^2 / 2) / n; the fractions are multinomial: a tail seen by few samples is a rough estimate of the energy.
    float variance = energy_body * energy_body * (variance_body + 0.5f * variance_body * variance_body) / max(body.w, 1.0f);
    float samples = max(body.w, 1.0f);
    if (has_bright)
    {
        variance += energy_bright * energy_bright * (variance_bright + 0.5f * variance_bright * variance_bright) / bright.w;
        samples += bright.w;
    }
    if (has_dark)
    {
        variance += energy_dark * energy_dark * (variance_dark + 0.5f * variance_dark * variance_dark) / dark.w;
        samples += dark.w;
    }
    variance += (square(first_bright - first_body) * p_bright * (1.0f - p_bright) + square(first_dark - first_body) * p_dark * (1.0f - p_dark)) / samples;
    RtLabEnergy result;
    result.log_energy = log(energy);
    result.relative_variance = variance / (energy * energy);
    return result;
}

// The energy test runs at this fraction of the distribution tolerance k: a donor whose energy is off by a factor must not get in, the energy is the thing
// the filter is there to protect. (Shader side tuning knob, no setting.)
#define RT_LAB_DISTRIBUTION_ENERGY_K_SCALE 0.5f
// A tap whose energy weight is below this is rejected without the other tests (the filter drops taps under 1e-6 anyway, this is the cheap side of that).
#define RT_LAB_DISTRIBUTION_EARLY_OUT 1e-4f

// The same energy estimate from an already built mixture (rt_lab_tail_mixture: the component first moments are in it, no extra exp).
func rt_lab_distribution_energy_of(RtLabTailMixture m) -> RtLabEnergy
{
    const float log_variance_body = (m.variance.x + 0.5f * m.variance.x * m.variance.x) / m.samples.x;
    float variance = m.energy.x * m.energy.x * log_variance_body;
    if (m.samples.y >= 1.0f)
    {
        const float log_variance_tail = (m.variance.y + 0.5f * m.variance.y * m.variance.y) / m.samples.y;
        const float tail_minus_body = m.first.y - m.first.x;
        variance += m.energy.y * m.energy.y * log_variance_tail + tail_minus_body * tail_minus_body * m.p * (1.0f - m.p) / m.count;
    }
    const float energy = max(m.mean_energy, 1e-6f);
    RtLabEnergy result;
    result.log_energy = log(energy);
    result.relative_variance = variance / (energy * energy);
    return result;
}

// Do two pixels share their energy? z = (ln E_c - ln E_t) against the uncertainty of both estimates (+ a 2 % floor), weight exp(-z^2 / (2 (s k)^2)), s =
// RT_LAB_DISTRIBUTION_ENERGY_K_SCALE. Few samples behind a tail widen the test, many samples make it strict. Identical states give 1.
func rt_lab_distribution_energy_weight(RtLabEnergy c, RtLabEnergy t, float k) -> float
{
    const float difference = c.log_energy - t.log_energy;
    const float noise = c.relative_variance + t.relative_variance + 4e-4f;
    const float scaled_k = k * RT_LAB_DISTRIBUTION_ENERGY_K_SCALE;
    return exp(-(difference * difference) / (2.0f * scaled_k * scaled_k * noise));
}

// The weight of a tap seen from a center for the body + bright tail mode. compare: RT_LAB_DISTRIBUTION_COMPARE_*. The center's mixture + energy are built
// ONCE per pixel by the caller (rt_lab_distribution_center), only the tap's are built per tap: this runs for every tap of every pass.
// Every comparison except BODY + TAIL alone also has to pass the total energy test (rt_lab_distribution_energy_weight): the tails are rare, they carry the
// energy, and the shape tests see a tail by its probability mass only (a shadowed pixel and a lit one with the same body and tails of 10 % against 30 % passed
// them at ~0.6). 0 DISTRIBUTION (default) = body + tail test AND energy test.
func rt_lab_distribution_compare_cached(int compare, RtLabTailMixture c, RtLabEnergy energy_c, float4 body_t, float4 tail_t, float k) -> float
{
    const RtLabTailMixture t = rt_lab_tail_mixture(body_t, tail_t);
    // The energy test first: it is the cheapest and rejects most of the taps across an edge, the shape tests are then skipped.
    const float energy_weight = rt_lab_distribution_energy_weight(energy_c, rt_lab_distribution_energy_of(t), k);
    if (energy_weight < RT_LAB_DISTRIBUTION_EARLY_OUT && compare != RT_LAB_DISTRIBUTION_COMPARE_BODY_TAIL)
    {
        return energy_weight;
    }
    float shape;
    switch (compare)
    {
        case RT_LAB_DISTRIBUTION_COMPARE_MEMBERSHIP: shape = rt_lab_distribution_compare_membership(c, t, k); break;
        case RT_LAB_DISTRIBUTION_COMPARE_PERCEPTUAL_MEAN: shape = rt_lab_distribution_compare_perceptual_mean(c, t, k); break;
        case RT_LAB_DISTRIBUTION_COMPARE_BODY_TAIL: return rt_lab_distribution_compare_body_tail(c, t, k);
        default: shape = rt_lab_distribution_compare_body_tail(c, t, k); break;
    }
    return min(shape, energy_weight);
}

// Same, building the center too (the live pixel probes: one pair per frame).
func rt_lab_distribution_compare(int compare, float4 body_c, float4 tail_c, float4 body_t, float4 tail_t, float k) -> float
{
    const RtLabTailMixture c = rt_lab_tail_mixture(body_c, tail_c);
    return rt_lab_distribution_compare_cached(compare, c, rt_lab_distribution_energy_of(c), body_t, tail_t, k);
}
