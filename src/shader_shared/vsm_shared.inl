#pragma once

#include "daxa/daxa.inl"
#include "globals.inl"
#include "scene.inl"

#define PAGE_ALIGN_AXIS_X 0
#define PAGE_ALIGN_AXIS_Z 2
// #define VSM_TEXTURE_RESOLUTION 2048
#define VSM_TEXTURE_RESOLUTION 4096
// #define VSM_TEXTURE_RESOLUTION 8192
#define VSM_MEMORY_RESOLUTION (8192)
#define VSM_PAGE_SIZE 128
#define VSM_CLIP_LEVELS 16
#define VSM_PAGE_TABLE_RESOLUTION (VSM_TEXTURE_RESOLUTION / VSM_PAGE_SIZE) // NOLINT(bugprone-integer-division)
#define VSM_INVALIDATE_PAGE_BLOCK_RESOLUTION 8
#define VSM_META_MEMORY_TABLE_RESOLUTION (VSM_MEMORY_RESOLUTION / VSM_PAGE_SIZE) // NOLINT(bugprone-integer-division)
#define VSM_PAGE_MASK_SIZE (((VSM_PAGE_TABLE_RESOLUTION * VSM_PAGE_TABLE_RESOLUTION) + 31) / 32)
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
    glmsf32mat4 point_light_projection_matrix;
    glmsf32mat4 inverse_point_light_projection_matrix;
};
DAXA_DECL_BUFFER_PTR(VSMGlobals)

struct VSMClipProjection
{
    daxa_i32vec2 page_offset;
    daxa_f32 near_to_far_range;
    daxa_f32 near_dist;
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
    // Is this page only being invalidated and requests redraw?
    daxa_u32 already_allocated;
    // TODO(msakmary) pack those ? move them into separate alloc request? idk...
    daxa_i32 point_light_index;
    daxa_i32 point_light_mip;
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
    daxa_u32 mask[VSM_PAGE_MASK_SIZE];
    daxa_i32vec2 clear_offset;
};
DAXA_DECL_BUFFER_PTR(FreeWrappedPagesInfo)

struct VSMPointLight
{
    daxa_ImageViewId page_table;
    glmsf32mat4 view_matrices[6];
    glmsf32mat4 inverse_view_matrices[6];
    daxa_BufferPtr(GPUPointLight) light;
};
DAXA_DECL_BUFFER_PTR_ALIGN(VSMPointLight, 8);