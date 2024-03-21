#pragma once

#include "daxa/daxa.inl"

#define VSM_TEXTURE_RESOLUTION 4096
#define VSM_MEMORY_RESULTION 4096
#define VSM_PAGE_SIZE 128
#define VSM_CLIP_LEVELS 4
#define VSM_PAGE_TABLE_RESOLUTION (VSM_TEXTURE_RESOLUTION / VSM_PAGE_SIZE)
#define VSM_META_MEMORY_RESOLUTION (VSM_MEMORY_RESOLUTION / VSM_PAGE_SIZE)

struct VSMClipProjection
{
    daxa_i32 height_offset;
    daxa_f32vec2 depth_page_offset;
    daxa_i32vec2 page_offset;
    daxa_f32mat4x4 view;
    daxa_f32mat4x4 projection;
    daxa_f32mat4x4 projection_view;
    daxa_f32mat4x4 inv_projection_view;
};
DAXA_DECL_BUFFER_PTR(VSMClipProjection)