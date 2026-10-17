#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#include "rt_lab_rays.inl"

// Coarse temporal anti-lag (RtLabSettings::coarse_anti_lag_*): variance clamping with two differences.
//   1. Massive range: the window comes from a heavily blurred, temporally accumulated 1/4 res estimate, so it is wide and
//      stable (no per pixel neighborhood noise).
//   2. No color clamp: the history COUNT is reduced instead, so the history adapts to the new value on its own.
// Passes (per frame, after the gather, before the temporal pass), all at gradient cell res (RT_LAB_GRADIENT_CELL^2 lab
// pixels per cell, each cell represented by the geometry of rt_lab_gradient_cell_pixel):
//   downsample  per cell, over ALL of the cell's rays this frame: (sum luminance, sum luminance^2, ray count)
//   blur H      edge aware gaussian (coarse tests, rt_lab_gradient_surface_weight) over the sums
//   blur V + temporal  same blur vertically -> this frame's blurred mean luminance, fed into an RTGI style coarse fast
//               history (fast mean + fast relative variance, firefly clamped) and a slow coarse mean (reprojected with
//               the previous camera); the RTGI anti-lag formula gives a length cap, stored as alpha = 1 / (cap + 1)
//               (cuts the coarse slow history too)
// The temporal pass reads alpha with an edge aware 2x2 upscale and caps the reprojected history length at 1 / alpha - 1
// (rt_lab_coarse_anti_lag_cap, rt_lab_shared.hlsl): the color is never touched. Luminance of the RTGI_RADIANCE_SCALE scaled
// radiance (unexposed).
#define RT_LAB_ANTI_LAG_X 8
#define RT_LAB_ANTI_LAG_Y 8
#define RT_LAB_ANTI_LAG_BLUR_PLANE_TOLERANCE 2.0f // blur + coarse reprojection: plane distance falloff in CELL widths
#define RT_LAB_ANTI_LAG_FAST_FIREFLY_STD 3.5f     // fast stats input clamp: fast mean x (1 + this x fast rel std), RTGI temporal_firefly_std_dev_clamp

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabAntiLagDownsampleH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, cell_sums) // .x sum lum, .y sum lum^2, .z rays
DAXA_DECL_TASK_HEAD_END

struct RtLabAntiLagDownsamplePush
{
    daxa_BufferPtr(RtLabAntiLagDownsampleH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // cells
    daxa_u32vec2 lab_size; // lab pixels
};

// horizontal: blur_src (cell sums) -> blur_dst (blurred sums)
// vertical:   blur_src (horizontal sums) -> coarse history (.x temporal mean luminance, .y clamp alpha (0 = no cut),
//             .z frames, .w the
//             cell's linear depth, 0 = sky / nothing), reprojecting history_previous
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabAntiLagBlurH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)                 // full res NDC depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)        // half res linear depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, blur_src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, blur_dst)          // horizontal pass output
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history_previous)  // vertical pass input
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, history)          // vertical pass output
// Vertical pass: RT_LAB_COARSE_ANTI_LAG_FRAME_MEAN / _RELATIVE_STD / _FRAME_RELATIVE_STD at cell res. Null in the horizontal pass.
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
// Vertical pass: the coarse fast history (.x fast mean luminance, .y fast relative variance, .z fast frames), reprojected
// with the same weights as the slow history.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, fast_history_previous)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, fast_history)
DAXA_DECL_TASK_HEAD_END

struct RtLabAntiLagBlurPush
{
    daxa_BufferPtr(RtLabAntiLagBlurH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // cells
    daxa_u32vec2 lab_size; // lab pixels
    daxa_u32 vertical;     // 0 = horizontal pass, 1 = vertical pass + temporal
};
