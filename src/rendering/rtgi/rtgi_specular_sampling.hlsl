#pragma once

// === RTGI Specular Ray Sampling ===================================================================
// Specular rays are drawn from the GGX distribution of visible normals (VNDF) around the half-res detail
// normal. On rough surfaces the lobe is wide enough that the pioneer guide (the same-frame bright-hit
// direction built for the diffuse rays, see rtgi_guided_sampling.hlsl) becomes useful as well, so a
// fraction of the rays is drawn from the guided disc warp instead. Both strategies are combined with the
// balance heuristic (one-sample MIS): weight = pdf_vndf / (p * pdf_guide + (1 - p) * pdf_vndf).
//
// The estimated quantity is the VNDF-weighted mean incoming radiance (the pre-filtered radiance of the split
// sum). Fresnel / visibility are applied at composite time via brdf_env_approx, so plain VNDF samples carry
// weight 1.
// #included by rtgi_trace_diffuse.hlsl.

#include "shader_lib/brdf.hlsl"
#include "rtgi_guided_sampling.hlsl"

// Keeps the view vector above the shading hemisphere (normal maps can tilt N away from the viewer).
func rtgi_specular_shading_normal(float3 detail_normal, float3 view_dir) -> float3
{
    const float NoV = dot(detail_normal, view_dir);
    return NoV < 0.02f ? normalize(detail_normal + view_dir * (0.02f - NoV)) : detail_normal;
}

// Rays that end up below the geometric surface would hit the surface itself. Fold them back just above the
// face plane (small bias, avoids black grazing reflections on normal-mapped surfaces).
func rtgi_specular_fold_above_face(float3 dir, float3 face_normal) -> float3
{
    const float d = dot(dir, face_normal);
    return d < 0.01f ? normalize(dir + face_normal * (0.01f - d)) : dir;
}

// VNDF reflection samples below the shading hemisphere (frequent at grazing view angles) used to get weight 0,
// i.e. a BLACK ray averaged into the pixel. But the signal is the lobe's mean incoming radiance and the energy
// lost to masking is applied at composite time (brdf_env_approx), so counting them as black darkens grazing
// reflections a second time (towards black at very shallow angles). With this on, such samples are redrawn up to
// RTGI_SPECULAR_BELOW_HORIZON_RETRIES times (= excluding them from the mean), and a last failure is folded just
// above the shading hemisphere instead of returning black. 0 = old behavior (weight 0).
#define RTGI_SPECULAR_RESAMPLE_BELOW_HORIZON 1
#define RTGI_SPECULAR_BELOW_HORIZON_RETRIES 4u

func rtgi_specular_guide_probability(float roughness, float guide_mix, bool guide_valid) -> float
{
    return guide_valid ? guide_mix * saturate((roughness - 0.25f) * (1.0f / 0.75f)) : 0.0f;
}

// Returns the world-space specular ray direction and its estimator weight (0 if the sample is unusable).
func rtgi_sample_specular_dir(
    float3 shading_normal,
    float3 face_normal,
    float3 view_dir,
    float roughness,
    float4 guide_sh_y,
    bool guide_valid,
    float guide_concentration,
    float guide_mix,
    out float weight) -> float3
{
    const float a = brdf_ggx_alpha(roughness);
    const float p_guide = rtgi_specular_guide_probability(roughness, guide_mix, guide_valid);

    float3x3 guide_basis = (float3x3)0;
    float2 guide_pd = float2(0, 0);
    float guide_kappa = 0.0f;
    if (p_guide > 0.0f)
    {
        guide_basis = rtgi_build_basis(shading_normal);
        guide_pd = rtgi_disc_project(guide_basis, rtgi_sh_dominant_direction(guide_sh_y, shading_normal));
        guide_kappa = rtgi_concentration_to_kappa(guide_concentration);
    }

    float3 dir;
    if (p_guide > 0.0f && rand() < p_guide)
    {
        dir = rtgi_sample_disc_warp_dir(guide_basis, guide_pd, guide_kappa);
    }
    else
    {
        dir = brdf_sample_ggx_vndf_reflection(shading_normal, view_dir, a, float2(rand(), rand()));
#if RTGI_SPECULAR_RESAMPLE_BELOW_HORIZON
        for (uint retry = 0u; retry < RTGI_SPECULAR_BELOW_HORIZON_RETRIES && dot(dir, shading_normal) <= 0.0f; ++retry)
        {
            dir = brdf_sample_ggx_vndf_reflection(shading_normal, view_dir, a, float2(rand(), rand()));
        }
        const float below = dot(dir, shading_normal);
        if (below <= 0.0f)
        {
            dir = normalize(dir + shading_normal * (0.01f - below)); // last resort: fold just above the horizon
        }
#endif
    }

    if (p_guide > 0.0f)
    {
        const float pdf_vndf = brdf_ggx_vndf_reflection_pdf(shading_normal, view_dir, dir, a);
        const float pdf_guide = rtgi_disc_warp_pdf(guide_basis, guide_pd, guide_kappa, dir);
        const float mix_pdf = p_guide * pdf_guide + (1.0f - p_guide) * pdf_vndf;
        weight = mix_pdf > 1e-8f ? pdf_vndf / mix_pdf : 0.0f;
    }
    else
    {
        weight = dot(dir, shading_normal) > 0.0f ? 1.0f : 0.0f;
    }

    return rtgi_specular_fold_above_face(dir, face_normal);
}
