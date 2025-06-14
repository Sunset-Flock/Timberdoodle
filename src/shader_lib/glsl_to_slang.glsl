#pragma once

#include "daxa/daxa.inl"

#if !defined(__cplusplus)
#if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)

// uint64_t imageAtomicMax(RWTexture2D<uint64_t> image, uint64_t value, int2 texel) {
//     uint64_t old = spirv_asm {
//         %ptrImageUlong = OpTypePointer Image $$uint64_t
//         %texelPointer = OpImageTexelPointer %ptrImageUlong $image $texel $(0)
//         result:$$uint64_t = OpAtomicUMax $$uint64_t %texelPointer $(0) $value
//     };
//     return old;
// }

#define findMSB firstbithigh

#define daxa_texture2D(TEX) Texture2D<float>::get(TEX)
#define daxa_utexture2DArray(TEX) Texture2DArray<uint>::get(TEX)
#define daxa_image2D(TEX) RWTexture2D<float>::get(TEX)
#define uintBitsToFloat(X) asfloat(X)
#define _mod(X, Y) ((X) - (Y) * floor((X)/(Y)))
#define _frac(X) frac(X)
#define mix(X, Y, Z) lerp(X, Y, Z)

float __texelFetch(Texture2D<float> tex, int2 index, uint mip)
{
    return tex.Load(uint3(index, mip));
}

uint _texelFetch(Texture2DArray<uint> tex, int3 index, uint mip)
{
    return tex.Load(uint4(index, mip));
}

void _imageStore(RWTexture2D<float> tex, int2 index, float value)
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

#else // #if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)

#define _mod(X, Y) mod(X,Y)
#define _frac(X) fract(X)
#define mul(M, V) ((M) * (V))
#define atan2(Y, X) atan(Y, X)

#endif // #else // #if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)
#endif // #if !defined(__cplusplus)