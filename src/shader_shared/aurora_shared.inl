#pragma once
#include "daxa/daxa.inl"

struct AuroraArcSegment
{
    daxa_f32vec3 s;  
    daxa_f32vec3 c1;  
    daxa_f32vec3 c2;  
    daxa_f32vec3 e;  
};
DAXA_DECL_BUFFER_PTR(AuroraArcSegment)
struct AuroraArc
{
    daxa_i32 segment_count;
    daxa_BufferPtr(AuroraArcSegment) arc_segments;
};
DAXA_DECL_BUFFER_PTR_ALIGN(AuroraArc, 8)

struct AuroraGlobals
{
    struct KernelInfo
    {
        daxa_i32 width;
        daxa_f32 variation;
    };
    daxa_f32vec2 start;
    daxa_f32vec2 end;
    daxa_f32vec3 B;
    daxa_f32 B_0; // Magnetic field intensity at height 0
    daxa_f32 height;
    daxa_f32 frequency;
    daxa_f32 phase_shift_per_layer;
    daxa_f32 offset_strength;
    daxa_f32 width;
    daxa_i32 layers;
    daxa_i32 beam_count;
    daxa_i32 beam_path_segment_count;
    daxa_f32 beam_path_length;
    daxa_f32 angle_offset_per_collision;
    daxa_u32 regenerate_aurora;
    daxa_u32 accumulate_aurora_luminance;
    daxa_u32vec2 aurora_image_resolution;
    KernelInfo rgb_blur_kernels[3];
    daxa_BufferPtr(daxa_f32vec3) emission_colors;
    daxa_BufferPtr(daxa_f32) emission_intensities;
#if defined(__cplusplus)
    AuroraGlobals()
        : start{-200.0f, 0.0f},
          end{200.0f, 100.0f},
          B{0.0, 0.01, -0.99},
          B_0{53'823},
          height{150.0f},
          frequency{3.05},
          phase_shift_per_layer{0.1},
          offset_strength{1.0f},
          width{5.0f},
          layers{30},
          beam_count{300'000},
          beam_path_segment_count{100},
          beam_path_length{40},
          angle_offset_per_collision{0.015},
          regenerate_aurora{1u},
          accumulate_aurora_luminance{1u},
          aurora_image_resolution{2560, 1440},
          rgb_blur_kernels{
              {.width = 29, .variation = 4.1f},
              {.width = 11, .variation = 2.7f},
              {.width = 5, .variation = 1.2f}}
    {
    }
#endif
};
DAXA_DECL_BUFFER_PTR_ALIGN(AuroraGlobals, 8)