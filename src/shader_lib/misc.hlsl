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

float IdFloatScramble(uint id)
{
    let SCRAMBLE = 0.172426234237f;
    return frac(float(id) * SCRAMBLE);
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

[ForceInline]
func AtomicAddU64(__ref uint64_t dest, uint64_t value) -> uint64_t
{
    uint64_t original_value;
    spirv_asm
    {
        OpCapability Int64Atomics;
        OpCapability Int64ImageEXT;
        %origin:$$uint64_t = OpAtomicIAdd &dest Device None $value;
        OpStore &original_value %origin
    };
    return original_value;
}

/// ===== From Shadertoy: https://www.shadertoy.com/view/llfcRl =====

uint packSnorm2x12(float2 v) 
{ 
    uint2 d = uint2(round(2047.5 + v*2047.5));
    return d.x|(d.y<<12u);
}

float2 unpackSnorm2x12(uint d) 
{
    return float2(uint2(d,d>>12)&4095u)/2047.5 - 1.0;
}

float msign( float v )
{
    return (v.x>=0.0) ? 1.0 : -1.0;
}

float2 msign2( float2 v )
{
    return float2( (v.x>=0.0) ? 1.0 : -1.0, 
                 (v.y>=0.0) ? 1.0 : -1.0 );
}

uint compress_normal_octahedral_24( in float3 nor )
{
    nor /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*msign2(nor.xy);
    return packSnorm2x12(nor.xy);
}

float3 uncompress_normal_octahedral_24( uint data )
{
    float2 v = unpackSnorm2x12(data);
    float3 nor = float3(v, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-nor.z,0.0);
    nor.x += (nor.x>0.0)?-t:t;
    nor.y += (nor.y>0.0)?-t:t;
    return normalize( nor );
}

uint compress_normal_octahedral_32( in float3 nor )
{
    nor.xy /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy  = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*msign2(nor.xy);
    uint2 d = uint2(round(32767.5 + nor.xy*32767.5));  
    return d.x|(d.y<<16u);
}

float3 uncompress_normal_octahedral_32( uint data )
{
    uint2 iv = uint2( data, data>>16u ) & 65535u; 
    float2 v = float2(iv)/32767.5 - 1.0;
    float3 nor = float3(v, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-nor.z,0.0);
    nor.x += (nor.x>0.0)?-t:t;
    nor.y += (nor.y>0.0)?-t:t;
    return normalize( nor );
}

float2 map_octahedral(float3 nor) {
    const float fac = 1.0f / (abs(nor.x) + abs(nor.y) + abs(nor.z));
    nor.x *= fac;
    nor.y *= fac;
    if (nor.z < 0.0f) {
        const float2 temp = nor.xy;
        nor.x = (1.0f - abs(temp.y)) * msign(temp.x);
        nor.y = (1.0f - abs(temp.x)) * msign(temp.y);
    }
    return float2(nor.x, nor.y) * 0.5f + 0.5f;
}

float3 unmap_octahedral(float2 v) {
    v = v * 2.0f - 1.0f;
    float3 nor = float3(v, 1.0f - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z, 0.0f);                        // much faster than original
    nor.x += (nor.x > 0.0f) ? -t : t;                   // implementation of this
    nor.y += (nor.y > 0.0f) ? -t : t;                   // technique
    return normalize(nor);
}

float interpolate_float(daxa_f32vec3 derivator, float v0, float v1, float v2)
{
    daxa_f32vec3 mergedV = daxa_f32vec3(v0, v1, v2);
    float ret;
    ret = dot(mergedV, derivator);
    return ret;
}

daxa_f32vec2 interpolate_vec2(daxa_f32vec3 derivator, daxa_f32vec2 v0, daxa_f32vec2 v1, daxa_f32vec2 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec2 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    return ret;
}

daxa_f32vec3 interpolate_vec3(daxa_f32vec3 derivator, daxa_f32vec3 v0, daxa_f32vec3 v1, daxa_f32vec3 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec3 merged_z = daxa_f32vec3(v0.z, v1.z, v2.z);
    daxa_f32vec3 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    return ret;
}

daxa_f32vec4 interpolate_vec4(daxa_f32vec3 derivator, daxa_f32vec4 v0, daxa_f32vec4 v1, daxa_f32vec4 v2)
{
    daxa_f32vec3 merged_x = daxa_f32vec3(v0.x, v1.x, v2.x);
    daxa_f32vec3 merged_y = daxa_f32vec3(v0.y, v1.y, v2.y);
    daxa_f32vec3 merged_z = daxa_f32vec3(v0.z, v1.z, v2.z);
    daxa_f32vec3 merged_w = daxa_f32vec3(v0.w, v1.w, v2.w);
    daxa_f32vec4 ret;
    ret.x = dot(merged_x, derivator);
    ret.y = dot(merged_y, derivator);
    ret.z = dot(merged_z, derivator);
    ret.w = dot(merged_w, derivator);
    return ret;
}

float square(float x)
{
    return x * x;
}

float2 square(float2 x)
{
    return x * x;
}

float3 square(float3 x)
{
    return x * x;
}

float4 square(float4 x)
{
    return x * x;
}

/// ===== =====