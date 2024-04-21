#pragma once

#include "daxa/daxa.inl"
#include "globals.inl"

// #define VSM_TEXTURE_RESOLUTION 2048
#define VSM_TEXTURE_RESOLUTION 4096
// #define VSM_TEXTURE_RESOLUTION 8192
#define VSM_MEMORY_RESOLUTION (8192)
#define VSM_PAGE_SIZE 128
#define VSM_CLIP_LEVELS 16
#define VSM_PAGE_TABLE_RESOLUTION (VSM_TEXTURE_RESOLUTION / VSM_PAGE_SIZE)
#define VSM_META_MEMORY_TABLE_RESOLUTION (VSM_MEMORY_RESOLUTION / VSM_PAGE_SIZE)

#define VSM_DEBUG_PAGE_TABLE_SCALE 10
#define VSM_DEBUG_PAGE_TABLE_RESOLUTION (VSM_PAGE_TABLE_RESOLUTION * VSM_DEBUG_PAGE_TABLE_SCALE)

#define VSM_DEBUG_META_MEMORY_TABLE_SCALE 10
#define VSM_DEBUG_META_MEMORY_TABLE_RESOLUTION (VSM_META_MEMORY_TABLE_RESOLUTION * VSM_DEBUG_META_MEMORY_TABLE_SCALE)

#define MAX_VSM_ALLOC_REQUESTS (512 * 512)
#if defined(__cplusplus)
// static_assert(VSM_PAGE_TABLE_RESOLUTION < 64, "VSM_PAGE_TABLE_RESOLUTION must be less than 64 or the dirty bit hiz must be extended");
#endif //defined(__cplusplus)

struct VSMGlobals
{
    daxa_f32 clip_0_texel_world_size;
    int force_clip_level;
};
DAXA_DECL_BUFFER_PTR(VSMGlobals)

struct VSMClipProjection
{
    daxa_i32 height_offset;
    daxa_f32vec2 depth_page_offset;
    daxa_i32vec2 page_offset;
    CameraInfo camera;
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