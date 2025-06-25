#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE 0
#define DEBUG_SHADER_DRAW_COORD_SPACE_NDC_MAIN_CAMERA 1
#define DEBUG_SHADER_DRAW_COORD_SPACE_NDC_VIEW_CAMERA 2

struct ShaderDebugLineDraw
{
    daxa_f32vec3 start;
    daxa_f32vec3 end;
    daxa_f32vec3 color;
    // 0 = worldspace, 1 = ndc.
    daxa_u32 coord_space;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugLineDraw)

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

struct ShaderDebugConeDraw
{
    daxa_f32vec3 position;
    daxa_f32vec3 direction;
    daxa_f32 size;
    daxa_f32 angle;
    daxa_u32 coord_space;
    daxa_f32vec3 color;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugConeDraw);

struct ShaderDebugSphereDraw
{
    daxa_f32vec3 position;
    daxa_f32 radius;
    daxa_u32 coord_space;
    daxa_f32vec3 color;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugSphereDraw);

struct ShaderDebugInput
{
    daxa_i32vec4 debug_ivec4;
    daxa_f32vec4 debug_fvec4;
};

struct ShaderDebugOutput
{
    daxa_i32vec4 debug_ivec4;
    daxa_f32vec4 debug_fvec4;
    daxa_u32 exceeded_line_draw_capacity;
    daxa_u32 exceeded_circle_draw_capacity;
    daxa_u32 exceeded_rectangle_draw_capacity;
    daxa_u32 exceeded_aabb_draw_capacity;
    daxa_u32 exceeded_box_draw_capacity;
    daxa_u32 exceeded_cone_draw_capacity;
    daxa_u32 exceeded_sphere_draw_capacity;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugOutput);

#define DebugDraws(DRAW_TYPE) DebugDraws_ ## DRAW_TYPE
#define DECL_DEBUG_DRAWS(DRAW_TYPE)\
struct DebugDraws_ ## DRAW_TYPE\
{\
    DrawIndirectStruct draw_indirect;\
    daxa_u32 draw_capacity;\
    daxa_u32 draw_requests;\
    daxa_RWBufferPtr(DRAW_TYPE) draws;\
};

DECL_DEBUG_DRAWS(ShaderDebugLineDraw)
DECL_DEBUG_DRAWS(ShaderDebugCircleDraw)
DECL_DEBUG_DRAWS(ShaderDebugRectangleDraw)
DECL_DEBUG_DRAWS(ShaderDebugAABBDraw)
DECL_DEBUG_DRAWS(ShaderDebugBoxDraw)
DECL_DEBUG_DRAWS(ShaderDebugConeDraw)
DECL_DEBUG_DRAWS(ShaderDebugSphereDraw)

struct ShaderDebugBufferHead
{
    DebugDraws(ShaderDebugLineDraw) line_draws;
    DebugDraws(ShaderDebugCircleDraw) circle_draws;
    DebugDraws(ShaderDebugRectangleDraw) rectangle_draws;
    DebugDraws(ShaderDebugAABBDraw) aabb_draws;
    DebugDraws(ShaderDebugBoxDraw) box_draws;
    DebugDraws(ShaderDebugConeDraw) cone_draws;
    DebugDraws(ShaderDebugSphereDraw) sphere_draws;
    ShaderDebugInput cpu_input;
    daxa_RWBufferPtr(ShaderDebugOutput) gpu_output;
};
DAXA_DECL_BUFFER_PTR(ShaderDebugBufferHead)

