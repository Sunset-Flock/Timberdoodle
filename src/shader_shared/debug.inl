#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE 0
#define DEBUG_SHADER_DRAW_COORD_SPACE_NDC 1
#define DEBUG_SHADER_DRAW_COORD_SPACE_NDC_OBSERVER 2

struct ShaderDebugCircleDraw
{
    daxa_f32vec3 position;
    daxa_f32vec3 color;
    daxa_f32 radius;
    // 0 = worldspace, 1 = ndc.
    daxa_u32 coord_space;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugCircleDraw)

struct ShaderDebugRectangleDraw
{
    daxa_f32vec3 center;
    daxa_f32vec3 color;
    daxa_f32vec2 span;
    // 0 = worldspace, 1 = ndc.
    daxa_u32 coord_space;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugRectangleDraw)

struct ShaderDebugAABBDraw
{
    daxa_f32vec3 position;
    daxa_f32vec3 size;
    daxa_f32vec3 color;
    daxa_u32 coord_space;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugAABBDraw)

struct ShaderDebugBoxDraw
{
    daxa_f32vec3 vertices [8];
    daxa_u32 coord_space;
    daxa_f32vec3 color;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugBoxDraw);

struct ShaderDebugLineDraw
{
    daxa_f32vec3 vertices[2];
    daxa_f32vec3 colors[2];
    daxa_u32 coord_space;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugLineDraw)

struct ShaderDebugInput
{
    daxa_i32vec2 texel_detector_pos;
    daxa_i32 texel_detector_window_half_size;
    daxa_i32vec4 debug_ivec4;
    daxa_f32vec4 debug_fvec4;
};

struct ShaderDebugOutput
{
    daxa_i32vec4 debug_ivec4;
    daxa_f32vec4 debug_fvec4;
    daxa_f32vec4 texel_detector_center_value;
    daxa_u32 exceeded_circle_draw_capacity;
    daxa_u32 exceeded_rectangle_draw_capacity;
    daxa_u32 exceeded_aabb_draw_capacity;
    daxa_u32 exceeded_box_draw_capacity;
    daxa_u32 exceeded_line_draw_capacity;
};

struct ShaderDebugBufferHead
{
    DrawIndirectStruct circle_draw_indirect_info;
    DrawIndirectStruct rectangle_draw_indirect_info;
    DrawIndirectStruct aabb_draw_indirect_info;
    DrawIndirectStruct box_draw_indirect_info;
    DrawIndirectStruct line_draw_indirect_info;
    daxa_u32 circle_draw_capacity;
    daxa_u32 rectangle_draw_capacity;
    daxa_u32 aabb_draw_capacity;
    daxa_u32 box_draw_capacity;
    daxa_u32 line_draw_capacity;
    ShaderDebugInput cpu_input;
    ShaderDebugOutput gpu_output;
    daxa_RWBufferPtr(ShaderDebugCircleDraw) circle_draws;
    daxa_RWBufferPtr(ShaderDebugRectangleDraw) rectangle_draws;
    daxa_RWBufferPtr(ShaderDebugAABBDraw) aabb_draws;
    daxa_RWBufferPtr(ShaderDebugBoxDraw) box_draws;
    daxa_RWBufferPtr(ShaderDebugLineDraw) line_draws;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugBufferHead)

