#pragma once

#include "daxa/daxa.inl"

struct CullData
{
    // Should equal (render_target_size + 1) / 2.
    // The mip level sizes are also ROUNDED UP from the previous size!
    // All pixels past the bounds are valid to read up to the `hiz_texture_size`.
    // All border pixels between `hiz_size` and `hiz_texture_size` are clamped values from the last valid pixel in the hiz.
    daxa_u32vec2 hiz_size;          
    daxa_f32vec2 hiz_size_rcp;          
    // Physical texture size is rounded up to next power of two.
    // This is done to simplify and speed up hiz generation.
    // Value used for bounds checks and uv calculations.
    daxa_u32vec2 physical_hiz_size;  
    daxa_f32vec2 physical_hiz_size_rcp;
};