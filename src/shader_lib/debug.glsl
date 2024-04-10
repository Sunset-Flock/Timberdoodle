#pragma once

#include "shader_shared/debug.inl"

#extension GL_EXT_debug_printf : enable

void debug_draw_circle(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugCircleDraw draw)
{
    const uint capacity = deref(debug_info).circle_draw_capacity;
    const uint index = atomicAdd(deref(debug_info).circle_draw_indirect_info.instance_count, 1);
    if (index < capacity)
    {
        deref_i(deref(debug_info).circle_draws, index) = draw;
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
        deref_i(deref(debug_info).rectangle_draws, index) = draw;
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
        deref_i(deref(debug_info).aabb_draws, index) = draw;
    }
    else
    {
        atomicAdd(deref(debug_info).gpu_output.exceeded_aabb_draw_capacity, 1);
    }
}

void debug_write_i32(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32vec2 xy, int value, int channel)
{
    if (all(equal(xy, deref(debug_info).cpu_input.texel_detector_pos)) && (channel >= 0) && (channel <= 4))
    {
        deref(debug_info).gpu_output.debug_ivec4[channel] = value;
    }
}

void debug_write_i32vec4(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32vec2 xy, daxa_i32vec4 value)
{
    if (all(equal(xy, deref(debug_info).cpu_input.texel_detector_pos)))
    {
        deref(debug_info).gpu_output.debug_ivec4 = value;
    }
}

void debug_write_f32(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32vec2 xy, float value, int channel)
{
    if (all(equal(xy, deref(debug_info).cpu_input.texel_detector_pos)) && (channel >= 0) && (channel <= 4))
    {
        deref(debug_info).gpu_output.debug_fvec4[channel] = value;
    }
}

void debug_write_f32vec4(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32vec2 xy, daxa_f32vec4 value)
{
    if (all(equal(xy, deref(debug_info).cpu_input.texel_detector_pos)))
    {
        deref(debug_info).gpu_output.debug_fvec4 = value;
    }
}

bool debug_in_lens(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32vec2 xy, out daxa_u32vec2 window_index)
{
    ShaderDebugInput cpu_in = deref(debug_info).cpu_input;
    daxa_u32vec2 window_top_left_corner = cpu_in.texel_detector_pos - cpu_in.texel_detector_window_half_size;
    daxa_u32vec2 window_bottom_right_corner = cpu_in.texel_detector_pos + cpu_in.texel_detector_window_half_size;
    window_index = xy - window_top_left_corner;
    return (all(greaterThanEqual(xy, window_top_left_corner)) && all(lessThanEqual(xy, window_bottom_right_corner)));
}

bool debug_in_lens_center(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32vec2 xy)
{
    return all(equal(xy, deref(debug_info).cpu_input.texel_detector_pos));
}

void debug_write_lens(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_ImageViewId debug_lens_image, daxa_u32vec2 xy, daxa_f32vec4 value)
{
    ShaderDebugInput cpu_in = deref(debug_info).cpu_input;
    daxa_u32vec2 window_index;
    if (debug_in_lens(debug_info, xy, window_index))
    {
#if (DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG)
        RWTexture2D<float>::get(debug_lens_image)[window_index] = value;
#else
        imageStore(daxa_image2D(debug_lens_image), daxa_i32vec2(window_index), value);
#endif

        if (all(equal(xy, cpu_in.texel_detector_pos)))
        {
            deref(debug_info).gpu_output.texel_detector_center_value = value;
        }
    }
}

#define DEBUG_INDEX(INDEX, MIN_INDEX, MAX_INDEX)                                                   \
    if (INDEX < MIN_INDEX || INDEX > MAX_INDEX)                                                    \
    {                                                                                              \
        debugPrintfEXT("index out of bounds: %i, range: [%i,%i]\n", INDEX, MIN_INDEX, MAX_INDEX);  \
    }                                                                                              