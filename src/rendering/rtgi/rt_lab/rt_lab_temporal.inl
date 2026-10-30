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
// SH guide input (null when no SH consumer is on): the diffuse as RTGI SH-Y + CoCg, accumulated with the same
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
// Perceptual luma moments (RtLabSettings::luma_moments_enabled, null when off), reprojected like the history and blended
// with the color's blend factor: (m1, m2, valid) of x = log(1 + luma / floor), center + distance weighted over 3x3 pixels.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, moments_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, moments)
// SH second moments (RT_LAB_SH_GUIDE_DIRECTIONAL_VARIANCE, null when off), 3x3 center + distance weighted per frame, reprojected + blended
// like the SH-Y (the first moment). A: (E[Y d.x^2], E[Y d.y^2], E[Y d.z^2], E[Y d.x d.y]), B: (E[Y d.x d.z], E[Y d.y d.z], E[Y^2], 0).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, sh_m2_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, sh_m2_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, sh_m2b_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, sh_m2b_accumulated)
// Neural training data (push.sq_bound, only while a neural dump runs): the per pixel (not blurred) mean of the squared ray Y, Co, Cg (.xyz, f32: Y is in
// RTGI_RADIANCE_SCALE units, its square overflows f16), reprojected + blended like the SH-Y (the first moments). .w unused.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, sq_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, sq_accumulated)
// Distribution guide (RtLabSettings::distribution_guide_mode, push.dist_mode, null when off): the per pixel body / tail mixture of
// x = log(1 + luma / floor), reprojected with the color's weights. A: (body mean, body variance, perceptual mean of all samples (the
// tail thresholds hang on it), body sample count), B: (tail fraction p, tail mean, tail variance, tail sample count). Body count < 1 = no state.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, dist_a_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dist_a)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, dist_b_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dist_b)
// Dark tail state C (mode RT_LAB_DISTRIBUTION_LOGNORMAL_3, null otherwise): (dark tail fraction, dark tail mean, dark tail variance, dark tail count).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, dist_c_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dist_c)
// COLOR accumulation step (push.color_pass, null in the statistics step): the first a-trous pass's output, the filtered frame. Its rgb replaces the raw
// mean as the new sample of the color history; the ray counts, the statistics and everything else still come from the raw rays.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_prefiltered)
DAXA_DECL_TASK_HEAD_END

struct RtLabTemporalPush
{
    daxa_BufferPtr(RtLabTemporalH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 split_enabled; // the split images are bound (graph built with split_history_enabled)
    daxa_u32 sh_enabled;    // the SH-Y + CoCg images are bound (graph built with an SH consumer)
    daxa_u32 gradient_bound; // temporal_gradient is bound
    daxa_u32 anti_lag_bound;    // coarse_anti_lag is bound
    daxa_u32 moments_bound;       // moments_history / moments are bound
    daxa_u32 sh_m2_bound;         // sh_m2_history / sh_m2_accumulated are bound
    daxa_u32 sq_bound;            // sq_history / sq_accumulated are bound
    daxa_u32 color_pass;          // 1 = the color accumulation step (lab_prefiltered is the new sample), 0 = the statistics step (the raw rays)
    daxa_u32 dist_mode;           // dist_* images are bound: RT_LAB_DISTRIBUTION_* (0 = not bound)
};

