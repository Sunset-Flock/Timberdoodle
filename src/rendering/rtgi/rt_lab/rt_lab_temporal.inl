#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#include "rt_lab_rays.inl"

#define RT_LAB_TEMPORAL_X 8
#define RT_LAB_TEMPORAL_Y 8

// Temporal accumulation (rt_lab_temporal.hlsl): reprojects the history with the previous camera, rejects taps by
// plane distance + face normal (disocclusion), blends in the frame's n rays (lab_raw_diffuse .a, ray redistribution) by
// n / history length. History .rgb = radiance * RTGI_RADIANCE_SCALE, .a = history length (samples = rays).
// Split history (rt_lab_settings.split_history_enabled, else the split images are null): two more histories in the same
// format, reprojected with the same weights, A only takes new samples on even frames, B on odd frames. |A - B| / 2
// estimates the error of the full history mean (each half holds half the samples: E[(A - B)^2] / 4 = sigma^2 / N).
// The pass also writes split_difference = ((lum(A) - lum(B)) / 2, valid) for the spatial passes + resolve to carry with
// the diffuse's weights (post-spatial error estimate), and draws the RT_LAB history / split A / B debug views.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabTemporalH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_raw_diffuse)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, split_a_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, split_b_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, split_a_accumulated)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, split_b_accumulated)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, split_difference)
// Accumulated per sample rgb variance (.rgb, RTGI_RADIANCE_SCALE^2 units), reprojected like the history: the 5x5 raw
// sample variance of every frame blended in by 1 / history length. Read by the spatial passes (variance guide).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, variance_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, variance)
// Coarse fallback input (null when the coarse fallback is off): the diffuse as RTGI SH-Y + CoCg, accumulated with the same
// reprojection weights (bilinear) and n / history length. cocg: .xy CoCg, .z history length. The frame's SH comes from
// the pixel's rays (ray_alloc, rays).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, sh_y_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, sh_y_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, cocg_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, cocg_accumulated)
// Temporal gradient lambda per cell (rt_lab_rays.inl), null when off: reprojected lengths x (1 - lambda).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, temporal_gradient)
// Coarse temporal anti-lag statistics per cell (rt_lab_anti_lag.inl), null when off: caps the reprojected lengths.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, coarse_anti_lag)
DAXA_DECL_TASK_HEAD_END

struct RtLabTemporalPush
{
    daxa_BufferPtr(RtLabTemporalH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 split_enabled; // the split images are bound (graph built with split_history_enabled)
    daxa_u32 sh_enabled;    // the SH-Y + CoCg images are bound (graph built with the coarse fallback)
    daxa_u32 gradient_bound; // temporal_gradient is bound
    daxa_u32 anti_lag_bound;    // coarse_anti_lag is bound
};
