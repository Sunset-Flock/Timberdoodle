#pragma once

#include "daxa/daxa.inl"

#if !defined(__cplusplus)
#if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)

#define findMSB firstbithigh

#define daxa_texture2D(TEX) Texture2D<float>::get(TEX)
#define daxa_utexture2DArray(TEX) Texture2DArray<uint>::get(TEX)
#define daxa_image2D(TEX) RWTexture2D<float>::get(TEX)
#define _mod(X, Y) (X - Y * floor(X/Y))
#define _frac(X) frac(X)

float4 texelFetch(Texture2D<float4> tex, int2 index, uint mip)
{
    return tex.Load(uint3(index, mip));
}

uint4 texelFetch(Texture2DArray<uint4> tex, int3 index, uint mip)
{
    return tex.Load(uint4(index, mip));
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
#define _frac(X) fract(X)
#define mul(M, V) (M * V)

#endif // #else // #if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)
#endif // #if !defined(__cplusplus)