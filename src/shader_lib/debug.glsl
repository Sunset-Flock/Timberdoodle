#pragma once

#include "shader_shared/debug.inl"

void draw_circle_ws(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugCircleDraw draw)
{
    const uint capacity = atomicAdd(deref(debug_info).circle_draw_capacity, 0);
    const uint index = atomicAdd(deref(debug_info).draw_indirect_info.instance_count, 1);
    if (index < capacity)
    {
        deref(deref(debug_info).circle_draws + index) = draw;
    }
}
