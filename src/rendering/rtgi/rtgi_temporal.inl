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
// Specular history sample count, reprojected with the surface motion: drives the specular ray demand.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, specular_frames_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, specular_sample_count)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_albedo_metalness) // ray demand visual impact
// Ray demand diffuse / specular share (rtgi_calc_ray_share): last frame's accumulated diffuse (SH-Y, .w luma) /
// specular history, nearest reprojected, times the material; written to ray_impact (.x diffuse, .y specular factor).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_diffuse_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_specular_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, ray_impact)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_normal_roughness) // pack_normal_roughness
// Bilinear coverage of OCCLUDER taps in the reprojection footprint (failed taps in front of the expected point),
// read by the specular accumulate's surface-motion footprint quality (RTGI_REPROJECT_OCCLUDER_AWARE_COUNT).
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, reproject_occluder_coverage)
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
// Specular channel (.rgb radiance, .a hit distance). Reprojected with the surface motion (shared corner +
// weights) and, for low roughness, with the virtual motion of the reflected image, which needs the current
// and previous half-res geometry.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, specular_pre_blurred)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, pre_filtered_specular_new)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, half_res_specular_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_specular_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, specular_frames_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, specular_frames_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, half_res_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_face_normals_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, specular_ray_count_image) // specular rays shot this frame
// Gloss + shading normal tests of the specular reprojection (current and previous frame).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_normal_roughness)          // pack_normal_roughness
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, half_res_normal_roughness_history)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RtgiRayCounters), ray_counters) // convergence statistics
// Specular fast history: .x fast brightness mean, .y fast relative variance, .z fast frame count (frames, not rays).
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, specular_fast_history_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, specular_fast_history_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, reproject_occluder_coverage) // written by reproject
DAXA_DECL_TASK_HEAD_END

struct RtgiTemporalAccumulatePush
{
    daxa_BufferPtr(RtgiTemporalAccumulateH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};