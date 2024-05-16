#pragma once
#include "daxa/daxa.inl"

struct AuroraGlobals
{
    daxa_f32vec2 start;
    daxa_f32vec2 end;
    daxa_f32vec3 B;
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
#if defined(__cplusplus)
    AuroraGlobals()
        : start{-100.0f, 0.0f},
          end{100.0f, 0.0f},
          B{0.0, 0.0, -1.0},
          height{30.0f},
          frequency{1},
          phase_shift_per_layer{0.1},
          offset_strength{1.0f},
          width{5.0f},
          layers{30},
          beam_count{3'000},
          beam_path_segment_count{100},
          beam_path_length{30},
          angle_offset_per_collision{0.05},
          regenerate_aurora{0u}
    {
    }
#endif
};
DAXA_DECL_BUFFER_PTR(AuroraGlobals)