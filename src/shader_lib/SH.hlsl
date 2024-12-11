
//=================================================================================================
//=================================================================================================
//=================FILE PORTED FROM HLSL2021 TO SLANG2024.3 BY IPOTRICK============================
//=================================================================================================
//=================================================================================================
//
//  SHforHLSL - Spherical harmonics suppport library for HLSL 2021, by MJP
//  https://github.com/TheRealMJP/SHforHLSL
//  https://therealmjp.github.io/
//
//  All code licensed under the MIT license
//
//=================================================================================================

//=================================================================================================
//
// This header is intended to be included directly from HLSL 2021+ code, or similar. It
// implements types and utility functions for working with low-order spherical harmonics,
// focused on use cases for graphics.
//
// Currently this library has support for L1 (2 bands, 4 coefficients) and
// L2 (3 bands, 9 coefficients) SH. Depending on the author and material you're reading, you may
// see L1 referred to as both first-order or second-order, and L2 referred to as second-order
// or third-order. Ravi Ramamoorthi tends to refer to three bands as second-order, and
// Peter-Pike Sloan tends to refer to three bands as third-order. This library always uses L1 and
// L2 for clarity.
//
// The core SH type as well as the L1 and L2 aliases are templated on the primitive scalar
// type (T) as well as the number of vector components (N). They are intended to be used with
// 1 or 3 components paired with the float32_t and float16_t primitive types (with
// -enable-16bit-types passed during compilation). When fp16 types are used, the helper functions
// take fp16 arguments to try to avoid implicit conversions where possible.
//
// The core SH type supports basic operator overloading for summing/subtracting two sets of SH
// coefficients as well as multiplying/dividing the set of SH coefficients by a value.
//
// Example #1: integrating and projecting radiance onto L2 SH
//
// SH::L2 radianceSH = SH::L2::Zero();
// for(int32_t sampleIndex = 0; sampleIndex < NumSamples; ++sampleIndex)
// {
//     float2 u1u2 = RandomFloat2(sampleIndex, NumSamples);
//     float3 sampleDirection = SampleDirectionSphere(u1u2);
//     float3 sampleRadiance = CalculateIncomingRadiance(sampleDirection);
//     radianceSH += SH::ProjectOntoL2(sampleDirection, sampleRadiance);
// }
// radianceSH *= 1.0f / (NumSamples * SampleDirectionSphere_PDF());
//
// Example #2: calculating diffuse lighting for a surface from radiance projected onto L2 SH
//
// SH::L2 radianceSH = FetchRadianceSH(surfacePosition);
// float3 diffuseLighting = SH::CalculateIrradiance(radianceSH, surfaceNormal) * (diffuseAlbedo / Pi);
//
//=================================================================================================

#pragma once

// Constants
static const float32_t Pi = 3.141592654f;
static const float32_t SqrtPi = sqrt(Pi);

static const float32_t CosineA0 = Pi;
static const float32_t CosineA1 = (2.0f * Pi) / 3.0f;
static const float32_t CosineA2 = (0.25f * Pi);

static const float32_t BasisL0 = 1 / (2 * SqrtPi);
static const float32_t BasisL1 = sqrt(3) / (2 * SqrtPi);
static const float32_t BasisL2_MN2 = sqrt(15) / (2 * SqrtPi);
static const float32_t BasisL2_MN1 = sqrt(15) / (2 * SqrtPi);
static const float32_t BasisL2_M0 = sqrt(5) / (4 * SqrtPi);
static const float32_t BasisL2_M1 = sqrt(15) / (2 * SqrtPi);
static const float32_t BasisL2_M2 = sqrt(15) / (4 * SqrtPi);

struct SH
{
    static const int32_t NumCoefficients = 9;

    half3 C[NumCoefficients];

    static SH Zero()
    {
        return (SH)0;
    }
};

SH operator+(SH a, SH other)
{
    SH result;
    [unroll]
    for(int32_t i = 0; i < a.NumCoefficients; ++i)
        result.C[i] = a.C[i] + other.C[i];
    return result;
}

SH operator-(SH a, SH other)
{
    SH result;
    [unroll]
    for(int32_t i = 0; i < a.NumCoefficients; ++i)
        result.C[i] = a.C[i] - other.C[i];
    return result;
}

SH operator*(SH a, half3 value)
{
    SH result;
    [unroll]
    for(int32_t i = 0; i < a.NumCoefficients; ++i)
        result.C[i] = a.C[i] * value;
    return result;
}

SH operator/(SH a, half3 value)
{
    SH result;
    [unroll]
    for(int32_t i = 0; i < a.NumCoefficients; ++i)
        result.C[i] = a.C[i] / value;
}

SH Lerp(SH x, SH y, half3 s)
{
    return x * (half3(1.0) - s) + y * s;
}

// Projects a value in a single direction onto a set of L2 SH coefficients
SH ProjectOntoL2(half3 direction, half3 value)
{
    SH sh;

    // L0
    sh.C[0] = half(BasisL0) * value;

    // L1
    sh.C[1] = half(BasisL1) * direction.y * value;
    sh.C[2] = half(BasisL1) * direction.z * value;
    sh.C[3] = half(BasisL1) * direction.x * value;

    // L2
    sh.C[4] = half(BasisL2_MN2) * direction.x * direction.y * value;
    sh.C[5] = half(BasisL2_MN1) * direction.y * direction.z * value;
    sh.C[6] = half(BasisL2_M0) * (half(3.0) * direction.z * direction.z - half(1.0)) * value;
    sh.C[7] = half(BasisL2_M1) * direction.x * direction.z * value;
    sh.C[8] = half(BasisL2_M2) * (direction.x * direction.x - direction.y * direction.y) * value;

    return sh;
}

// Calculates the dot product of two sets of L2 SH coefficients
half3 DotProduct(SH a, SH b)
{
    half3 result = half3(0.0);
    for(int32_t i = 0; i < SH::NumCoefficients; ++i)
        result += a.C[i] * b.C[i];

    return result;
}

// Projects a delta in a direction onto SH and calculates the dot product with a set of L2 SH coefficients.
// Can be used to "look up" a value from SH coefficients in a particular direction.
half3 Evaluate(SH sh, half3 direction)
{
    SH projectedDelta = ProjectOntoL2(direction, (half3)(1.0));
    return DotProduct(projectedDelta, sh);
}

// Convolves a set of L2 SH coefficients with a set of L2 zonal harmonics
SH ConvolveWithZH(SH sh, half3 zh)
{
    // L0
    sh.C[0] *= zh.x;

    // L1
    sh.C[1] *= zh.y;
    sh.C[2] *= zh.y;
    sh.C[3] *= zh.y;

    // L2
    sh.C[4] *= zh.z;
    sh.C[5] *= zh.z;
    sh.C[6] *= zh.z;
    sh.C[7] *= zh.z;
    sh.C[8] *= zh.z;

    return sh;
}

// Convolves a set of L2 SH coefficients with a cosine lobe. See [2]
SH ConvolveWithCosineLobe(SH sh)
{
    return ConvolveWithZH(sh, half3(CosineA0, CosineA1, CosineA2));
}

// Calculates the irradiance from a set of SH coefficients containing projected radiance.
// Convolves the radiance with a cosine lobe, and then evaluates the result in the given normal direction.
// Note that this does not scale the irradiance by 1 / Pi: if using this result for Lambertian diffuse,
// you will want to include the divide-by-pi that's part of the Lambertian BRDF.
// For example: float3 diffuse = CalculateIrradiance(sh, normal) * diffuseAlbedo / Pi;
half3 CalculateIrradiance(SH sh, half3 normal)
{
    SH convolved = ConvolveWithCosineLobe(sh);
    return Evaluate(convolved, normal);
}

// Approximates a GGX lobe with a given roughness/alpha as L2 zonal harmonics, using a fitted curve
half3 ApproximateGGXAsL2ZH(half ggxAlpha)
{
    const half l1Scale = half(1.66711256633276) / (half(1.65715038133932) + ggxAlpha);
    const half l2Scale = half(1.56127990596116) / (half(0.96989757593282) + ggxAlpha) - half(0.599972342361123);
    return half3(1.0, l1Scale, l2Scale);
}

// Convolves a set of L2 SH coefficients with a GGX lobe for a given roughness/alpha
SH ConvolveWithGGX(SH sh, half ggxAlpha)
{
    return ConvolveWithZH(sh, ApproximateGGXAsL2ZH(ggxAlpha));
}


// References:
//
// [0] Stupid SH Tricks by Peter-Pike Sloan - https://www.ppsloan.org/publications/StupidSH36.pdf
// [1] Converting SH Radiance to Irradiance by Graham Hazel - https://grahamhazel.com/blog/2017/12/22/converting-sh-radiance-to-irradiance/
// [2] An Efficient Representation for Irradiance Environment Maps by Ravi Ramamoorthi and Pat Hanrahan - https://cseweb.ucsd.edu/~ravir/6998/papers/envmap.pdf
// [3] SHMath by Chuck Walbourn (originally written by Peter-Pike Sloan) - https://walbourn.github.io/spherical-harmonics-math/
// [4] ZH3: Quadratic Zonal Harmonics by Thomas Roughton, Peter-Pike Sloan, Ari Silvennoinen, Michal Iwanicki, and Peter Shirley - https://torust.me/ZH3.pdf
// [5] Precomputed Global Illumination in Frostbite by Yuriy O'Donnell - https://www.ea.com/frostbite/news/precomputed-global-illumination-in-frostbite