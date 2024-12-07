#pragma once

#extension GL_EXT_debug_printf : enable

#include "shader_shared/debug.inl"

#define GPU_ASSERT_STRING "GPU ASSERT FAILED IN \"" __FILE__ "\": "

#define GPU_ASSERTS 1

void debug_draw_line(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugCircleDraw draw)
{

}

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
#if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)
        RWTexture2D<float4>::get(debug_lens_image)[window_index] = value;
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