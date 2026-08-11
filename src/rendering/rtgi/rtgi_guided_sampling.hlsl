#pragma once

// === RTGI Pioneer Guided Ray Sampling ============================================================
// Bends diffuse ray directions toward the guide direction built by the same-frame pioneer trace +
// spatial RIS resample (rtgi_guide_resample.hlsl), instead of sampling a plain cosine hemisphere.
// #included by rtgi_trace_diffuse.hlsl.
//
// MALLEY'S METHOD
// A uniform point (x,y) on the unit disc, lifted to (x,y,sqrt(1-x^2-y^2)), is a cosine-weighted
// sample of the hemisphere around `normal`. That lift is what plain cosine sampling already does.
//
// BENDING TOWARD THE GUIDE
// Project the guide direction onto that same disc -- call the projected point `pd`
// (rtgi_disc_project). Instead of drawing the disc sample uniformly, pull it toward `pd` first
// (rtgi_sample_disc_warp_dir), then lift as usual. The pull is clamped to the disc's own boundary in
// whichever direction it's heading (rtgi_disc_boundary_t_max), so the bent point never leaves the
// disc -- which means the lifted direction never leaves the true hemisphere either, unlike a lobe
// built directly on the sphere around the guide axis (that can point past the horizon once the guide
// is oblique to the normal).
//
// CONCENTRATION
// rtgi_settings.guide_concentration ([0,1]) is the one knob for how hard the bend is, same value for
// every pixel and every guiding mode. 0 = no bend, i.e. exactly plain cosine sampling. 1 = every
// sample collapses onto `pd`. The pull itself is a Schlick-style rational falloff
// (rtgi_schlick_pull) on the normalized distance from `pd`.
// Typically good values are between 0.85f and 0.95f.
//
// WEIGHT
// Bending moves the sampling density away from cosine, so cos(theta)/PI no longer cancels against it
// for free. rtgi_sample_guided_diffuse_dir returns weight = pdf_cosine/pdf_guide alongside the
// direction; the caller multiplies the traced radiance by it to stay unbiased.
//
// Used from rtgi_trace_diffuse.hlsl, gated by rtgi_settings.pioneer_guiding_enabled -- the guide_sh_y
// passed in comes from the pioneer trace + spatial RIS resample (rtgi_guide_resample.hlsl).

#include "shader_shared/shared.inl"
#include "shader_lib/misc.hlsl"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

static const float RTGI_GUIDE_INV_PI = 1.0f / PI;

// Epsilon-biased helper axis keeps the cross product non-degenerate when `axis` is exactly +Z (same
// trick as the inline TBN build in rtgi_trace_diffuse.hlsl).
func rtgi_build_basis(float3 axis) -> float3x3
{
    const float3 tangent = normalize(cross(axis, float3(0, 0, 1) + 0.0001f));
    const float3 bitangent = cross(tangent, axis);
    // Columns = tangent, bitangent, axis -- mul(basis, local) maps local +Z to world `axis`.
    return transpose(float3x3(tangent, bitangent, axis));
}

// sh_y is a single RIS-picked (hit_direction * Y, Y) pair (see entry_guide_resolve in
// rtgi_guide_resample.hlsl), never averaged -- dividing back out Y recovers that pick's direction.
// Falls back to the surface normal when no candidate was found this frame (sh_y.xyz ~= 0, e.g. sky or
// no matching-surface pioneer cell nearby) -- harmless to bend toward, just not yet useful.
func rtgi_sh_dominant_direction(float4 sh_y, float3 normal_fallback) -> float3
{
    const float moment_len = length(sh_y.xyz);
    return moment_len > 1e-8f ? (sh_y.xyz / moment_len) : normal_fallback;
}

// concentration -> von Mises-Fisher kappa, standard closed-form approximation for S^2 (Banerjee et
// al.): kappa ~= R*(3-R^2)/(1-R^2). rc=min(...,0.999) is a float safety margin, not a cap --
// concentration=1 would give kappa=Infinity, not a bigger-but-finite bend.
func rtgi_concentration_to_kappa(float concentration) -> float
{
    const float rc = clamp(concentration, 0.0f, 0.999f);
    return rc * (3.0f - rc * rc) / max(1.0f - rc * rc, 1e-4f);
}

// No pow/log/exp -- cheaper than a power warp even though power "looks" simpler (pow() for a
// non-integer exponent is usually exp2(c*log2(x)) under the hood anyway).
func rtgi_schlick_pull(float u, float c) -> float
{
    return u / (1.0f + c * (1.0f - u));
}

func rtgi_schlick_pull_pdf(float u_prime, float c) -> float
{
    const float d = 1.0f + c * u_prime;
    return (1.0f + c) / (d * d);
}

// basis is orthonormal, so its transpose is its inverse: world direction -> local (x,y) on the disc.
func rtgi_disc_project(float3x3 basis, float3 dir) -> float2
{
    return mul(transpose(basis), dir).xy;
}

// Ray/circle intersection from an interior point: |pd + t*dir2| = 1, solved for t>0. pd generally
// isn't the disc's center, so this varies by direction.
func rtgi_disc_boundary_t_max(float2 dir2, float2 pd) -> float
{
    const float b = dot(pd, dir2);
    const float c = 1.0f - dot(pd, pd);
    return -b + sqrt(max(0.0f, b * b + c));
}

// theta is NOT sampled uniformly. A genuinely uniform disc point, re-expressed in polar coordinates
// around an off-center pole pd, does not have a uniform azimuth marginal -- directions with more room
// before the rim carry more probability. Sampling theta flat-uniform would skew even c=0 toward
// whichever side pd sits on. Fix: derive theta from a real origin-centered uniform disc point p0
// instead -- p0's own (theta, s0^2), where s0 is the normalized distance from pd to p0, has s0^2
// exactly Uniform[0,1) and independent of theta, for any pd. Bending only s0^2 and leaving theta as
// p0 gave it means c=0 reconstructs p0 exactly.
func rtgi_sample_disc_warp_dir(float3x3 basis, float2 pd, float c) -> float3
{
    const float u1 = rand();
    const float u2 = rand();
    const float r0 = sqrt(u1);
    const float phi0 = 2.0f * PI * u2;
    const float2 p0 = float2(r0 * cos(phi0), r0 * sin(phi0));

    const float2 delta = p0 - pd;
    const float rho0 = length(delta);
    const float theta = atan2(delta.y, delta.x);
    const float t_max = rtgi_disc_boundary_t_max(float2(cos(theta), sin(theta)), pd);
    const float s0_sq = t_max > 1e-6f ? min((rho0 / t_max) * (rho0 / t_max), 1.0f - 1e-6f) : 0.0f;

    const float s = sqrt(rtgi_schlick_pull(s0_sq, c));
    const float rho = s * t_max;
    const float2 p = pd + rho * float2(cos(theta), sin(theta));
    const float z = sqrt(max(0.0f, 1.0f - dot(p, p)));
    return mul(basis, float3(p, z));
}

// p(w) = f_disc(x,y) * cos(theta), f_disc(x,y) = pdf_schlick(s^2) / pi -- no t_max(theta) term,
// because theta's own (t_max(theta)^2-weighted) marginal is undisturbed by the bend (see the sampler
// above) and exactly cancels the rho<->s^2 Jacobian. Returns 0 outside the true hemisphere.
func rtgi_disc_warp_pdf(float3x3 basis, float2 pd, float c, float3 dir) -> float
{
    const float3 local = mul(transpose(basis), dir);
    if (local.z <= 0.0f) { return 0.0f; }

    const float2 delta = local.xy - pd;
    const float rho = length(delta);
    const float2 dir2 = rho > 1e-6f ? delta / rho : float2(1.0f, 0.0f);
    const float t_max = rtgi_disc_boundary_t_max(dir2, pd);
    const float s2 = min((rho / t_max) * (rho / t_max), 1.0f - 1e-6f);

    return rtgi_schlick_pull_pdf(s2, c) / PI * local.z;
}

func rtgi_cosine_hemi_pdf(float3 normal, float3 dir) -> float
{
    return max(dot(normal, dir), 0.0f) * RTGI_GUIDE_INV_PI;
}

// CALLER MUST multiply the traced radiance by `weight` before accumulating it. At concentration=0,
// pdf_guide reduces exactly to pdf_cosine, so weight == 1 always -- zero code-path difference from
// plain cosine sampling.
func rtgi_sample_guided_diffuse_dir(
    float3 normal,
    float4 guide_sh_y,
    float concentration,
    out float weight
) -> float3
{
    const float3 guide_axis = rtgi_sh_dominant_direction(guide_sh_y, normal);
    const float c = rtgi_concentration_to_kappa(concentration);

    // Built once and reused for both sampling and pdf evaluation below -- rtgi_disc_warp_pdf needs
    // `dir` expressed in this same frame.
    const float3x3 normal_basis = rtgi_build_basis(normal);
    const float2 pd = rtgi_disc_project(normal_basis, guide_axis);

    const float3 dir = rtgi_sample_disc_warp_dir(normal_basis, pd, c);

    const float pdf_cosine = rtgi_cosine_hemi_pdf(normal, dir);
    const float pdf_guide = rtgi_disc_warp_pdf(normal_basis, pd, c, dir);

    weight = pdf_guide > 1e-8f ? (pdf_cosine / pdf_guide) : 0.0f;

    return dir;
}
