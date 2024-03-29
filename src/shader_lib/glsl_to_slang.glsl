#pragma once

#include "daxa/daxa.inl"

#if !defined(__cplusplus)
#if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)

#define findMSB firstbithigh

#define daxa_texture2D(TEX) daxa_Texture2D(float4, TEX)
#define daxa_image2D(TEX) daxa_RWTexture2D(float4, TEX)
#define _mod(X, Y) fmod(X,Y)

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
    uint prev;
    InterlockedAdd((*dst), value, prev);
    return prev;
}
#define atomicAdd(DST_LVAL, VALUE) _atomicAdd(&(DST_LVAL), VALUE)

uint _atomicMax(uint* dst, uint value)
{
    uint prev;
    InterlockedMax((*dst), value, prev);
    return prev;
}
#define atomicMax(DST_LVAL, VALUE) _atomicMax(&(DST_LVAL), VALUE)

#define greaterThan(X, Y) ((X) > (Y))
#define lessThan(X, Y) ((X) < (Y))
#define lessThanEqual(X, Y) ((X) <= (Y))
#define greaterThanEqual(X, Y) ((X) >= (Y))
#define equal(X, Y) ((X) == (Y))

#else // #if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)

#define _mod(X, Y) mod(X,Y)
#define mul(M, V) (M * V)

#endif // #else // #if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)
#endif // #if !defined(__cplusplus)