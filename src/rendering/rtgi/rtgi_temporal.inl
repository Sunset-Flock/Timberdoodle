#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_TEMPORAL_X 8
#define RTGI_TEMPORAL_Y 8

// === Temporal Reprojection ===
// Computes where this pixel's history lives and how valid it is, without touching the color/statistics
// history. Outputs the addressing + weights so the accumulation pass (and future pre-trace consumers)
// can read history cheaply:
//   - half_res_sample_count : final accumulated sample count (<0 == sky, 0 == disocclusion)
//   - reproject_corner      : (bilinear.origin + 1) as u16x2, gather-uv = corner * inv_size
//   - reproject_weights     : bilinear custom weights (occlusion*normal folded in), unorm8x4
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiTemporalReprojectH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DIndex<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_normal)                  // probably best to use face or smooth normal here
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, half_res_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_normal_history)          // probably best to use face or smooth normal here
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_sample_count_history)      // packed: normal count (10b) + fast frames (6b)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, half_res_sample_count)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec2>, reproject_corner)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, reproject_weights)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RtgiRayCounters), ray_counters)
DAXA_DECL_TASK_HEAD_END

struct RtgiTemporalReprojectPush
{
    daxa_BufferPtr(RtgiTemporalReprojectH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};

// === Temporal Accumulation ===
// Consumes the reprojection metadata to read color/statistics history and blend it with the new frame.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiTemporalAccumulateH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DIndex<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DIndex<daxa_u32>, half_res_sample_count)      // packed normal+fast; normal <0 == sky, 0 == disocclusion
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_sample_count_history)     // previous frame packed counters (for fast-history reprojection)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, ray_count_image)                  // rays shot this frame (R8_UINT, from trace/allocate)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, reproject_corner)             // (bilinear.origin + 1) as u16x2
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, reproject_weights)            // bilinear custom weights (unorm8x4)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_diffuse_pre_blurred)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, pre_filtered_diffuse_new)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, pre_filtered_diffuse2_new)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, half_res_diffuse_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_diffuse_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, half_res_diffuse2_pre_blurred)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, half_res_diffuse2_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, half_res_diffuse2_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, fast_temporal_history_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, fast_temporal_history_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, ao_guide_new)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, half_res_ao_guide_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, half_res_ao_guide_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, perceptual_radiance_new)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, temporal_perceptual_radiance_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, temporal_perceptual_radiance_history)
DAXA_DECL_TASK_HEAD_END

struct RtgiTemporalAccumulatePush
{
    daxa_BufferPtr(RtgiTemporalAccumulateH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};