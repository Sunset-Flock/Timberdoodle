#pragma once

#include "daxa/daxa.inl"

// x: texel index x
// y: texel index y
// z: linearzed depth
// return: dithered colored depth value that bands less
daxa_f32vec3 unband_z_color(int x, int y, float z)
{
    const float dither_increment = (1.0 / 256.0 * 0.25);
    float dither = dither_increment * 0.25 + dither_increment * ((int(x) % 2) + 2 * (int(y) % 2));
    daxa_f32vec3 color = daxa_f32vec3(1,0.66666,0.33333) * z + dither;
    return color;
}

// assumes infinite far plane
// assumes inverse z
// depth: depth
// near: near plane
// return: linear depth
float linearise_depth(float depth, float near)
{
    return near / (depth);
}

// x: texel index x
// y: texel index y
// depth: non-linear inverse depth
// near: near plane
// far: NOT far place, but the distance at which the red channel reaches 1.0
// depth: non linear inverse depth
// return: dithered colored depth value that bands less
daxa_f32vec3 unband_depth_color(int x, int y, float depth, float near, float far)
{
    const float dither_increment = (1.0 / 256.0) * 0.25;
    float dither = dither_increment * 0.25 + dither_increment * ((int(x) % 2) + 2 * (int(y) % 2));
    daxa_f32vec3 color = daxa_f32vec3(1.0,0.66666,0.33333) * linearise_depth(depth, near) * 1.0 / (far) + dither;
    return color;
}

daxa_f32 MM_Hash2(daxa_f32vec2 v)
{
  return _frac(1e4 * sin(17.0 * v.x + 0.1 * v.y) * (0.1 + abs(sin(13.0 * v.y + v.x))));
}

daxa_f32 MM_Hash3(daxa_f32vec3 v)
{
  return MM_Hash2(daxa_f32vec2(MM_Hash2(v.xy), v.z));
}

// Hashed Alpha Testing
// https://casual-effects.com/research/Wyman2017Hashed/Wyman2017Hashed.pdf
// maxObjSpaceDerivLen = max(length(dFdx(i_objectSpacePos)), length(dFdy(i_objectSpacePos)));
daxa_f32 compute_hashed_alpha_threshold(daxa_f32vec3 object_space_pos, daxa_f32 max_obj_space_deriv_len, daxa_f32 hash_scale)
{
    daxa_f32 pix_scale = 1.0 / (hash_scale + max_obj_space_deriv_len);
    daxa_f32 pix_scale_min = exp2(floor(log2(pix_scale)));
    daxa_f32 pix_scale_max = exp2(ceil(log2(pix_scale)));
    daxa_f32vec2 alpha = daxa_f32vec2(MM_Hash3(floor(pix_scale_min * object_space_pos)), MM_Hash3(floor(pix_scale_max * object_space_pos)));
    daxa_f32 lerp_factor = _frac(log2(pix_scale));
    daxa_f32 x = (1.0 - lerp_factor) * alpha.x + lerp_factor * alpha.y;
    daxa_f32 a = min(lerp_factor, 1.0 - lerp_factor);
    daxa_f32vec3 cases = daxa_f32vec3(x * x / (2.0 * a * (1.0 - a)), (x - 0.5 * a) / (1.0 - a), 1.0 - ((1.0 - x) * (1.0 - x) / (2.0 * a * (1.0 - a))));
    daxa_f32 threshold = (x < (1.0 - a)) ? ((x < a) ? cases.x : cases.y) : cases.z;
    return clamp(threshold, 1e-6, 1.0);
}