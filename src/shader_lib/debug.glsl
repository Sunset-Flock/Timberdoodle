#pragma once

#include "shader_shared/debug.inl"

void draw_circle_ws(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugCircleDraw draw)
{
    const uint capacity = deref(debug_info).circle_draw_capacity;
    const uint index = atomicAdd(deref(debug_info).circle_draw_indirect_info.instance_count, 1);
    if (index < capacity)
    {
        deref(deref(debug_info).circle_draws + index) = draw;
    }
}

void draw_rectangle_ws(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugRectangleDraw draw)
{
    const uint capacity = deref(debug_info).rectangle_draw_capacity;
    const uint index = atomicAdd(deref(debug_info).rectangle_draw_indirect_info.instance_count, 1);
    if (index < capacity)
    {
        deref(deref(debug_info).rectangle_draws + index) = draw;
    }
}
