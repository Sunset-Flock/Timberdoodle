#pragma once

#include <daxa/daxa.inl>
#include "shared.inl"

struct ShaderDebugCircleDraw
{
    daxa_f32vec2 position;
    daxa_f32 radius;
};

struct ShaderDebugBufferHead
{
    daxa_u32 circle_draw_capacity;
    daxa_u32 circle_draw_count;
    daxa_RWBufferPtr(DebugCircleDraw) circle_draws;
};