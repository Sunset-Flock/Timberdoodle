#pragma once

#include <daxa/daxa.inl>

// x: texel index x
// y: texel index y
// z: linearzed depth
// return: dithered colored depth value that bands less
vec3 unband_z_color(int x, int y, float z)
{
    const float dither_increment = (1.0 / 256.0 * 0.25);
    float dither = dither_increment * 0.25 + dither_increment * ((int(x) % 2) + 2 * (int(y) % 2));
    vec3 color = vec3(1,0.66666,0.33333) * z + dither;
    return color;
}

// assumes infinite far plane
// assumes inverse z
// depth: depth
// near: near plane
// return: linear depth
float linearisze_depth(float depth, float near)
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
vec3 unband_depth_color(int x, int y, float depth, float near, float far)
{
    const float dither_increment = (1.0 / 256.0) * 0.25;
    float dither = dither_increment * 0.25 + dither_increment * ((int(x) % 2) + 2 * (int(y) % 2));
    vec3 color = vec3(1.0,0.66666,0.33333) * linearisze_depth(depth, near) * 1.0 / (far) + dither;
    return color;
}