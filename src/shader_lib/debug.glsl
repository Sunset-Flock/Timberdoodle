#pragma once

#include "shader_shared/debug.inl"

void debug_draw_circle(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugCircleDraw draw)
{
    const uint capacity = deref(debug_info).circle_draw_capacity;
    const uint index = atomicAdd(deref(debug_info).circle_draw_indirect_info.instance_count, 1);
    if (index < capacity)
    {
        deref(deref(debug_info).circle_draws + index) = draw;
    }
    else
    {
        atomicAdd(deref(debug_info).gpu_output.exceeded_circle_draw_capacity, 1);
    }
}

void debug_draw_rectangle(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugRectangleDraw draw)
{
    const uint capacity = deref(debug_info).rectangle_draw_capacity;
    const uint index = atomicAdd(deref(debug_info).rectangle_draw_indirect_info.instance_count, 1);
    if (index < capacity)
    {
        deref(deref(debug_info).rectangle_draws + index) = draw;
    }
    else
    {
        atomicAdd(deref(debug_info).gpu_output.exceeded_rectangle_draw_capacity, 1);
    }
}

void debug_draw_aabb(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugAABBDraw draw)
{
    const uint capacity = deref(debug_info).aabb_draw_capacity;
    const uint index = atomicAdd(deref(debug_info).aabb_draw_indirect_info.instance_count, 1);
    if (index < capacity)
    {
        deref(deref(debug_info).aabb_draws + index) = draw;
    }
    else
    {
        atomicAdd(deref(debug_info).gpu_output.exceeded_aabb_draw_capacity, 1);
    }
}

void debug_under_cursor_write_i32(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, uvec2 xy, int value, int channel)
{
    if (xy == deref(debug_info).cpu_input.texel_detector_pos && channel >= 0 && channel <= 4)
    {
        deref(debug_info).gpu_output.debug_ivec4[channel] = value;
    }
}

void debug_under_cursor_write_f32(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, uvec2 xy, float value, int channel)
{
    if (xy == deref(debug_info).cpu_input.texel_detector_pos && channel >= 0 && channel <= 4)
    {
        deref(debug_info).gpu_output.debug_fvec4[channel] = value;
    }
}