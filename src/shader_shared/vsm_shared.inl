#pragma once

#include "daxa/daxa.inl"
#include "globals.inl"
#include "scene.inl"

#define PAGE_ALIGN_AXIS_X 0
#define PAGE_ALIGN_AXIS_Z 2
// #define VSM_TEXTURE_RESOLUTION 2048
#define VSM_TEXTURE_RESOLUTION 4096
// #define VSM_TEXTURE_RESOLUTION 8192
#define VSM_MEMORY_RESOLUTION (8192 * 2)
#define VSM_PAGE_SIZE 64
#define VSM_CLIP_LEVELS 16

// #define VSM_PAGE_TABLE_RESOLUTION (VSM_TEXTURE_RESOLUTION / VSM_PAGE_SIZE) // NOLINT(bugprone-integer-division)
#define VSM_PAGE_TABLE_RESOLUTION (VSM_TEXTURE_RESOLUTION / VSM_PAGE_SIZE) // NOLINT(bugprone-integer-division)
#define VSM_INVALIDATE_PAGE_BLOCK_RESOLUTION 8
#define VSM_META_MEMORY_TABLE_RESOLUTION (VSM_MEMORY_RESOLUTION / VSM_PAGE_SIZE) // NOLINT(bugprone-integer-division)
#define VSM_PAGE_MASK_SIZE (((VSM_PAGE_TABLE_RESOLUTION * VSM_PAGE_TABLE_RESOLUTION) + 31) / 32)
#define VSM_DEBUG_PAGE_TABLE_SCALE 10
#define VSM_DEBUG_PAGE_TABLE_RESOLUTION (VSM_PAGE_TABLE_RESOLUTION * VSM_DEBUG_PAGE_TABLE_SCALE)

#define VSM_DEBUG_META_MEMORY_TABLE_SCALE 10
#define VSM_DEBUG_META_MEMORY_TABLE_RESOLUTION (VSM_META_MEMORY_TABLE_RESOLUTION * VSM_DEBUG_META_MEMORY_TABLE_SCALE)

// #define VSM_POINT_LIGHT_NEAR 0.001f
#define VSM_POINT_LIGHT_NEAR 0.6f

#define MAX_VSM_ALLOC_REQUESTS (512 * 512)
#if defined(__cplusplus)
static_assert(VSM_PAGE_TABLE_RESOLUTION <= 64, "VSM_PAGE_TABLE_RESOLUTION must be less than 64 or the dirty bit hiz must be extended");
static_assert(VSM_PAGE_TABLE_RESOLUTION <= (1u << 7), "VSM_PAGE_TABLE_RESOLUTION must be less than 2^8 because of coord packing into meta memory");

// TODO: Pack point light shit further so that we can fit more here
static_assert(MAX_POINT_LIGHTS <= 32, "MAX_POINT_LIGHTS must be less than 128 because of packing into meta entry");
static_assert(VSM_TEXTURE_RESOLUTION == 4096, "Point lights require this right now - need to adjust mip count");
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

struct AllocationRequest
{
    daxa_i32vec3 coords;
    // Is this page only being invalidated and requests redraw?
    daxa_u32 already_allocated;
    // TODO(msakmary) pack those ? move them into separate alloc request? idk...
    daxa_i32 point_light_index;
    daxa_i32 point_light_mip;
};
struct VSMAllocationRequestsHeader
{
    daxa_u32 counter;
    AllocationRequest requests[MAX_VSM_ALLOC_REQUESTS];
};
DAXA_DECL_BUFFER_PTR(VSMAllocationRequestsHeader)

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
    CameraInfo face_cameras[6];
    daxa_BufferPtr(GPUPointLight) light;
};
DAXA_DECL_BUFFER_PTR_ALIGN(VSMPointLight, 8);