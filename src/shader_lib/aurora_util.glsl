#pragma once

#include <daxa/daxa.inl>
#include "shader_shared/aurora_shared.inl"
#include "shader_lib/glsl_to_slang.glsl"

daxa_i32 beam_segment_buffer_idx(daxa_i32 beam_idx, daxa_i32 segment_idx, daxa_BufferPtr(AuroraGlobals) globals)
{
    return beam_idx * globals.beam_path_segment_count + segment_idx;
}