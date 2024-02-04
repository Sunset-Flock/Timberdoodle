#pragma once

#include "../debug.inl"

void draw_circle_ss(daxa_RWBufferPtr(DebugInfo) debug_info, vec2 uv_position, float uv_radius, DebugCircleDraw draw)
{
    const uint capacity = atomicAdd(deref(debug_info).circle_draw_capacity, 0);
    const uint index = atomicAdd(deref(debug_info).circle_draw_count, 1);
    if (index < capacity)
    {
        deref(deref(debug_info).circle_draws + index) = draw;
    }
}
