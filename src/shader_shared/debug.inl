#pragma once

#include <daxa/daxa.inl>
#include "shared.inl"

struct ShaderDebugCircleDraw
{
    daxa_f32vec3 position;
    daxa_f32 radius;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugCircleDraw)

struct ShaderDebugBufferHead
{
    DrawIndirectStruct draw_indirect_info;
    daxa_u32 circle_draw_capacity;
    daxa_RWBufferPtr(ShaderDebugCircleDraw) circle_draws;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugBufferHead)