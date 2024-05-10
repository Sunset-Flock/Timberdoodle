#pragma once

#include "daxa/daxa.inl"

[[vk::binding(DAXA_STORAGE_IMAGE_BINDING, 0)]] RWTexture2D<daxa::u64> tex_rw_u64_table[];
[[vk::binding(DAXA_STORAGE_IMAGE_BINDING, 0)]] RWTexture2D<daxa::u32> RWTexture2D_utable[];

func firstbitlow_uint4(uint4 v) -> uint
{
    uint vec_mask = 
        (v[0] > 0 ? 1u << 0u : 0u) |
        (v[1] > 0 ? 1u << 1u : 0u) |
        (v[2] > 0 ? 1u << 2u : 0u) |
        (v[3] > 0 ? 1u << 3u : 0u);
    uint first_scalar_with_bits = firstbitlow(vec_mask);
    return firstbitlow(v[first_scalar_with_bits]);
}

func firstbithigh_uint4(uint4 v) -> uint
{
    uint vec_mask = 
        (v[0] > 0 ? 1u << 0u : 0u) |
        (v[1] > 0 ? 1u << 1u : 0u) |
        (v[2] > 0 ? 1u << 2u : 0u) |
        (v[3] > 0 ? 1u << 3u : 0u);
    uint first_scalar_with_bits = firstbithigh(vec_mask);
    return firstbithigh(v[first_scalar_with_bits]);
}

// Copyright 2019 Google LLC.
// SPDX-License-Identifier: Apache-2.0

// Polynomial approximation in GLSL for the Turbo colormap
// Original LUT: https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f

// Authors:
//   Colormap Design: Anton Mikhailov (mikhailov@google.com)
//   GLSL Approximation: Ruofei Du (ruofei@google.com)

float3 TurboColormap(float x)
{
    const float4 kRedVec4 = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const float4 kGreenVec4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const float4 kBlueVec4 = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const float2 kRedVec2 = float2(-152.94239396, 59.28637943);
    const float2 kGreenVec2 = float2(4.27729857, 2.82956604);
    const float2 kBlueVec2 = float2(-89.90310912, 27.34824973);

    x = clamp(x, 0, 1);
    float4 v4 = float4( 1.0, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;
    return float3(
      dot(v4, kRedVec4)   + dot(v2, kRedVec2),
      dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
      dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    );
}

float3 hsv2rgb(float3 c) {
    float4 k = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * lerp(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

static uint _rand_state;
void rand_seed(uint seed) {
    _rand_state = seed;
}

float rand() {
    // https://www.pcg-random.org/
    _rand_state = _rand_state * 747796405u + 2891336453u;
    uint result = ((_rand_state >> ((_rand_state >> 28u) + 4u)) ^ _rand_state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0;
}

[ForceInline]
func AtomicMaxU64(__ref uint64_t dest, uint64_t value) -> uint64_t
{
    uint64_t original_value;
    spirv_asm
    {
        OpCapability Int64Atomics;
        %origin:$$uint64_t = OpAtomicUMax &dest Device None $value;
        OpStore &original_value %origin
    };
    return original_value;
}