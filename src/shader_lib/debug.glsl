#pragma once

#extension GL_EXT_debug_printf : enable

#include "shader_shared/debug.inl"

#define GPU_ASSERT_ENABLE 1

#if GPU_ASSERT_ENABLE

static bool _GPU_ASSERT_FAIL_BOOL = false;
#define GPU_ASSERT_FAIL _GPU_ASSERT_FAIL_BOOL

#define GPU_ASSERT(COND)\
{\
    if (!COND)\
    {\
        printf("GPU ASSERT FAILED IN " __FILE__ ":%i: " #COND "\n", __LINE__);\
        GPU_ASSERT_FAIL = true;\
    }\
}

#define GPU_ASSERT_COMPARE_INT(A, OP, B)\
{\
    int a = A;\
    int b = B;\
    bool COND = a OP b;\
    if (!COND)\
    {\
        printf("GPU ASSERT FAILED IN " __FILE__ ":%i: " #A " " #OP " " #B ": %i " #OP " %i\n", __LINE__, a, b);\
        GPU_ASSERT_FAIL = true;\
    }\
}

#else

#define GPU_ASSERT_FAIL false
#define GPU_ASSERT(COND)
#define GPU_ASSERT_COMPARE_INT(A, OP, B)

#endif

#define DEBUG_DRAW(DEBUG_DATA, field, VALUE)\
{\
    const uint capacity = deref(DEBUG_DATA).field ## _draws.draw_capacity;\
    const uint index = atomicAdd(deref(DEBUG_DATA).field ## _draws.draw_requests, 1);\
    if (index < capacity)\
    {\
        atomicAdd(deref(DEBUG_DATA).field ## _draws.draw_indirect.instance_count, 1);\
        deref_i(deref(DEBUG_DATA).field ## _draws.draws, index) = draw;\
    }\
    else\
    {\
        atomicAdd(deref(DEBUG_DATA).gpu_output.exceeded_ ## field ## _draw_capacity, 1);\
    }\
}

void debug_draw_line(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugLineDraw draw)
{
    DEBUG_DRAW(debug_info, line, draw)
}

daxa_u32 debug_alloc_line_draws(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32 amount)
{
    const uint capacity = deref(debug_info).line_draws.draw_capacity;
    const uint offset = atomicAdd(deref(debug_info).line_draws.draw_requests, amount);
    if ((offset + amount) < capacity)
    {
        atomicAdd(deref(debug_info).line_draws.draw_indirect.instance_count, amount);
        return offset;
    }
    return ~0u;
}

void debug_draw_circle(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugCircleDraw draw)
{
    DEBUG_DRAW(debug_info, circle, draw)
}

void debug_draw_rectangle(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugRectangleDraw draw)
{
    DEBUG_DRAW(debug_info, rectangle, draw)
}

void debug_draw_aabb(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugAABBDraw draw)
{
    DEBUG_DRAW(debug_info, aabb, draw)
}

void debug_draw_box(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugBoxDraw draw)
{
    DEBUG_DRAW(debug_info, box, draw)
}

void debug_draw_cone(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugConeDraw draw)
{
    DEBUG_DRAW(debug_info, cone, draw)
}

void debug_draw_sphere(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugSphereDraw draw)
{
    DEBUG_DRAW(debug_info, sphere, draw)
}

#define DEBUG_INDEX(INDEX, MIN_INDEX, MAX_INDEX)                                                   \
    if (INDEX < MIN_INDEX || INDEX > MAX_INDEX)                                                    \
    {                                                                                              \
        debugPrintfEXT("index out of bounds: %i, range: [%i,%i]\n", INDEX, MIN_INDEX, MAX_INDEX);  \
    }                                                                                              