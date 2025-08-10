#pragma once

#include <daxa/daxa.inl>

#define DEBUG_DRAW_CLONE_X 16
#define DEBUG_DRAW_CLONE_Y 16
struct DebugTaskDrawDebugDisplayPush
{
    daxa::ImageViewId src;
    daxa::RWTexture2DIndex<daxa_f32vec4> dst;
    daxa_u32vec2 src_size;
    daxa::u32 image_view_type;
    daxa_i32 format;
    daxa::f32 float_min;
    daxa::f32 float_max;
    daxa::i32 int_min;
    daxa::i32 int_max;
    daxa::u32 uint_min;
    daxa::u32 uint_max;
    daxa::i32 rainbow_ints;
    daxa_i32vec4 enabled_channels;
    daxa_i32vec2 mouse_over_index;
    daxa_BufferPtr(daxa_f32vec4) readback_ptr;
    daxa_u32 readback_index;
};