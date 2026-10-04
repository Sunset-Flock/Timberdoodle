#pragma once

// GGX / Smith microfacet helpers shared by direct shading (shade_opaque) and the RTGI specular trace.
// Conventions: `roughness` is the perceptual (glTF) roughness in [0,1], `a` = roughness^2 is the GGX alpha.
// All directions are unit vectors pointing AWAY from the surface (V = towards the eye, L = towards the light).

#ifndef BRDF_PI
#define BRDF_PI 3.1415926535897932384626433832795
#endif

// Lowest GGX alpha used anywhere. Keeps D() finite for perfect mirrors (roughness 0).
static const float BRDF_MIN_GGX_ALPHA = 0.0016f; // == roughness 0.04

func brdf_ggx_alpha(float roughness) -> float
{
    return max(roughness * roughness, BRDF_MIN_GGX_ALPHA);
}

// Dielectric F0 of 4% blended towards the albedo by metalness (glTF metallic-roughness model).
func brdf_specular_f0(float3 albedo, float metalness) -> float3
{
    return lerp(float3(0.04f, 0.04f, 0.04f), albedo, metalness);
}

func brdf_diffuse_color(float3 albedo, float metalness) -> float3
{
    return albedo * (1.0f - metalness);
}

func brdf_fresnel_schlick(float3 f0, float VoH) -> float3
{
    const float f = pow(1.0f - saturate(VoH), 5.0f);
    return f0 + (1.0f - f0) * f;
}

func brdf_ggx_d(float NoH, float a) -> float
{
    const float a2 = a * a;
    const float d = (NoH * a2 - NoH) * NoH + 1.0f;
    return a2 / (BRDF_PI * d * d);
}

// Smith G1 for GGX (separable form).
func brdf_smith_g1(float NoX, float a) -> float
{
    const float a2 = a * a;
    const float NoX2 = NoX * NoX;
    return 2.0f * NoX / (NoX + sqrt(a2 + (1.0f - a2) * NoX2));
}

// Height-correlated Smith visibility term V = G2 / (4 NoL NoV).
func brdf_smith_ggx_visibility(float NoV, float NoL, float a) -> float
{
    const float a2 = a * a;
    const float ggx_v = NoL * sqrt(NoV * NoV * (1.0f - a2) + a2);
    const float ggx_l = NoV * sqrt(NoL * NoL * (1.0f - a2) + a2);
    return 0.5f / max(ggx_v + ggx_l, 1e-8f);
}

// Cook-Torrance GGX specular BRDF times NoL (radiance response to a unit-irradiance directional light).
func brdf_ggx_specular_nol(float3 N, float3 V, float3 L, float3 f0, float roughness) -> float3
{
    const float NoL = dot(N, L);
    const float NoV = max(dot(N, V), 1e-4f);
    if (NoL <= 0.0f) { return float3(0, 0, 0); }
    const float3 H = normalize(V + L);
    const float NoH = saturate(dot(N, H));
    const float VoH = saturate(dot(V, H));
    const float a = brdf_ggx_alpha(roughness);
    return brdf_ggx_d(NoH, a) * brdf_smith_ggx_visibility(NoV, NoL, a) * brdf_fresnel_schlick(f0, VoH) * NoL;
}

// Analytic fit of the split-sum environment BRDF (Karis, "Physically Based Shading on Mobile").
// Integral of the GGX specular lobe (incl. Fresnel + visibility) over the hemisphere for uniform incoming
// radiance. Multiply pre-filtered (lobe-averaged) incoming radiance by this to get outgoing specular.
func brdf_env_approx(float3 f0, float roughness, float NoV) -> float3
{
    const float4 c0 = float4(-1.0f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = float4(1.0f, 0.0425f, 1.04f, -0.04f);
    const float4 r = roughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28f * saturate(NoV))) * r.x + r.y;
    const float2 AB = float2(-1.04f, 1.04f) * a004 + r.zw;
    return f0 * AB.x + AB.y;
}

func brdf_build_tbn(float3 n) -> float3x3
{
    // Branchless orthonormal basis (Duff et al. 2017). Rows = tangent, bitangent, normal.
    const float s = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z);
    const float b = n.x * n.y * a;
    const float3 t = float3(1.0f + s * n.x * n.x * a, s * b, -s * n.x);
    const float3 bt = float3(b, s + n.y * n.y * a, -n.y);
    return float3x3(t, bt, n);
}

// Samples a GGX microfacet normal from the distribution of VISIBLE normals (VNDF) with the spherical cap
// method (Dupuy & Benyoub 2023). Isotropic. All vectors in the local frame where the surface normal is +Z.
func brdf_sample_ggx_vndf_local(float3 V_local, float a, float2 u) -> float3
{
    // Warp to the hemisphere configuration.
    const float3 Vh = normalize(float3(a * V_local.x, a * V_local.y, V_local.z));
    // Sample a spherical cap in (-Vh.z, 1].
    const float phi = 2.0f * BRDF_PI * u.x;
    const float z = (1.0f - u.y) * (1.0f + Vh.z) - Vh.z;
    const float sin_theta = sqrt(saturate(1.0f - z * z));
    const float3 c = float3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    // Halfway direction in the hemisphere configuration, then unwarp.
    const float3 Hh = c + Vh;
    return normalize(float3(a * Hh.x, a * Hh.y, max(Hh.z, 0.0f)));
}

// World-space VNDF reflection sample. Returns the reflected direction L.
func brdf_sample_ggx_vndf_reflection(float3 N, float3 V, float a, float2 u) -> float3
{
    const float3x3 tbn = brdf_build_tbn(N);
    const float3 V_local = mul(tbn, V);
    const float3 H_local = brdf_sample_ggx_vndf_local(V_local, a, u);
    const float3 H = mul(transpose(tbn), H_local);
    return reflect(-V, H);
}

// Solid-angle pdf of brdf_sample_ggx_vndf_reflection producing L: D(H) * G1(V) / (4 * NoV).
func brdf_ggx_vndf_reflection_pdf(float3 N, float3 V, float3 L, float a) -> float
{
    const float NoV = max(dot(N, V), 1e-4f);
    if (dot(N, L) <= 0.0f) { return 0.0f; }
    const float3 H = normalize(V + L);
    const float NoH = saturate(dot(N, H));
    return brdf_ggx_d(NoH, a) * brdf_smith_g1(NoV, a) / (4.0f * NoV);
}
