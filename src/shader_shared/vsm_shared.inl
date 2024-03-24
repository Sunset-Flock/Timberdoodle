#pragma once

#include "daxa/daxa.inl"

#define VSM_TEXTURE_RESOLUTION 4096
#define VSM_MEMORY_RESOLUTION 4096
#define VSM_PAGE_SIZE 128
#define VSM_CLIP_LEVELS 4
#define VSM_PAGE_TABLE_RESOLUTION (VSM_TEXTURE_RESOLUTION / VSM_PAGE_SIZE)
#define VSM_META_MEMORY_TABLE_RESOLUTION (VSM_MEMORY_RESOLUTION / VSM_PAGE_SIZE)

#define VSM_DEBUG_PAGE_TABLE_SCALE 16
#define VSM_DEBUG_PAGE_TABLE_RESOLUTION (VSM_PAGE_TABLE_RESOLUTION * VSM_DEBUG_PAGE_TABLE_SCALE)

#define VSM_DEBUG_META_MEMORY_TABLE_SCALE 3
#define VSM_DEBUG_META_MEMORY_TABLE_RESOLUTION (VSM_META_MEMORY_TABLE_RESOLUTION * VSM_DEBUG_META_MEMORY_TABLE_SCALE)

#define MAX_VSM_ALLOC_REQUESTS 256

struct VSMGlobals
{
    daxa_f32 clip_0_texel_world_size;
};
DAXA_DECL_BUFFER_PTR(VSMGlobals)

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

struct AllocationCount
{
    daxa_u32 count;
};
DAXA_DECL_BUFFER_PTR(AllocationCount)

struct AllocationRequest
{
    daxa_i32vec3 coords;
};
DAXA_DECL_BUFFER_PTR(AllocationRequest)

struct PageCoordBuffer
{
    daxa_i32vec2 coords;
};
DAXA_DECL_BUFFER_PTR(PageCoordBuffer)

struct FindFreePagesHeader
{
    daxa_u32 free_buffer_counter;
    daxa_u32 not_visited_buffer_counter;
};
DAXA_DECL_BUFFER_PTR(FindFreePagesHeader)

struct FreeWrappedPagesInfo
{
    daxa_i32vec2 clear_offset;
};
DAXA_DECL_BUFFER_PTR(FreeWrappedPagesInfo)