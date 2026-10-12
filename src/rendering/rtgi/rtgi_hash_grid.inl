#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_HASH_GRID_QUERY_X 8
#define RTGI_HASH_GRID_QUERY_Y 8
#define RTGI_HASH_GRID_CELL_PASS_X 64
#define RTGI_HASH_GRID_DEBUG_X 8
#define RTGI_HASH_GRID_DEBUG_Y 8

// The hash grid runs in four passes. The query runs before the temporal reproject, the other three after the
// temporal accumulate (they read its output):
//   query    (per pixel)    : find / insert the cell, pixel_count atomic add -> the pixel's index within its cell
//   allocate (per cell slot): age / free untouched cells; touched cells reserve a pixel list segment with ONE fused
//                             64 bit atomic (segment index + list offset)
//   scatter  (per pixel)    : write the pixel into its cell's segment at list_offset + index within cell
//   blend    (per segment)  : set the cell to the plain mean of its pixels' accumulated diffuse

// === Query ===
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiHashGridQueryH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtgiHashGridCell), hash_grid)
// .x cell index (RTGI_HASH_GRID_INVALID_CELL = none), .y index of the pixel within the cell's list
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec2>, pixel_cell)
DAXA_DECL_TASK_HEAD_END

struct RtgiHashGridQueryPush
{
    daxa_BufferPtr(RtgiHashGridQueryH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};

// === Allocate ===
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiHashGridAllocateH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtgiHashGridCell), hash_grid)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtgiHashGridListHeader), list_header) // cleared every frame
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), segment_cells)                  // segment -> cell slot
DAXA_DECL_TASK_HEAD_END

struct RtgiHashGridAllocatePush
{
    daxa_BufferPtr(RtgiHashGridAllocateH::AttachmentShaderBlob) attach;
};

// === Scatter ===
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiHashGridScatterH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, pixel_cell)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiHashGridCell), hash_grid)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), pixel_list) // packed half-res pixel xy, one entry per queried pixel
DAXA_DECL_TASK_HEAD_END

struct RtgiHashGridScatterPush
{
    daxa_BufferPtr(RtgiHashGridScatterH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};

// === Blend ===
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiHashGridBlendH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, diffuse)   // temporally accumulated SH-Y
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, diffuse2)  // temporally accumulated CoCg
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, sample_count)  // accumulate's packed counts (rtgi_pack_sample_counts)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtgiHashGridCell), hash_grid)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiHashGridListHeader), list_header)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), segment_cells)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), pixel_list)
DAXA_DECL_TASK_HEAD_END

struct RtgiHashGridBlendPush
{
    daxa_BufferPtr(RtgiHashGridBlendH::AttachmentShaderBlob) attach;
};

// === Hash grid debug visualization ===
// Full res: looks up the cell of every pixel (no insert) and draws the DEBUG_DRAW_MODE_RTGI_HASH_GRID_* views.
// Only dispatched while one of those modes is active.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiHashGridDebugH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DIndex<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_camera_detail_normal_image)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiHashGridCell), hash_grid)
DAXA_DECL_TASK_HEAD_END

struct RtgiHashGridDebugPush
{
    daxa_BufferPtr(RtgiHashGridDebugH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};
