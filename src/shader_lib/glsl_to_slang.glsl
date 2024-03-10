#pragma once

/// --- glsl to slang begin ---

#include "daxa/daxa.inl"

#if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)

#define findMSB firstbithigh

#define daxa_texture2D(TEX) daxa_Texture2D(float4, TEX)
#define daxa_image2D(TEX) daxa_RWTexture2D(float4, TEX)

float4 texelFetch(Texture2D<float4> tex, uint2 index, uint mip)
{
    return tex.Load(uint3(index, mip));
}

void imageStore(RWTexture2D<float4> tex, int2 index, float4 value)
{
    tex[index] = value;
}

uint _atomicAdd(uint * dst, uint value)
{
    uint out;
    InterlockedAdd((*dst), value, out);
    return out;
}
#define atomicAdd(DST_LVAL, VALUE) _atomicAdd(&(DST_LVAL), VALUE)

uint _atomicMax(uint* dst, uint value)
{
    uint out;
    InterlockedMax((*dst), value, out);
    return out;
}
#define atomicMax(DST_LVAL, VALUE) _atomicMax(&(DST_LVAL), VALUE)

#define greaterThan(X, Y) ((X) > (Y))
#define lessThan(X, Y) ((X) < (Y))
#define lessThanEqual(X, Y) ((X) <= (Y))
#define greaterThanEqual(X, Y) ((X) >= (Y))
#define equal(X, Y) ((X) == (Y))

#else // #if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)

#define mul(M, V) (M * V)

#endif // #else // #if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)

/// --- glsl to slang end ---