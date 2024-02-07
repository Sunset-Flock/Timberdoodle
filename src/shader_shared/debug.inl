#pragma once

#include <daxa/daxa.inl>
#include "shared.inl"

#define DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE 0
#define DEBUG_SHADER_DRAW_COORD_SPACE_NDC 1

struct ShaderDebugCircleDraw
{
    daxa_f32vec3 position;
    daxa_f32vec3 color;
    daxa_f32 radius;
    // 0 = worldspace, 1 = ndc.
    daxa_u32 coord_space;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugCircleDraw)

struct ShaderDebugBufferHead
{
    DrawIndirectStruct draw_indirect_info;
    daxa_u32 circle_draw_capacity;
    daxa_RWBufferPtr(ShaderDebugCircleDraw) circle_draws;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugBufferHead)