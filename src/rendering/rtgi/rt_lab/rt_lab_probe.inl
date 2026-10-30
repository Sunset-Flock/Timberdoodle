#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

// Live pixel probes (rt_lab_probe.hpp): every frame the denoiser state of up to RT_LAB_PROBE_COUNT pixels is copied into a small host
// visible ring buffer (one slot per frame) and shown in the UI. Words (u32 / bit cast f32) of a slot:
//   header (RT_LAB_PROBE_HEADER_WORDS):  0 stamp (never 0)   1 active mask   2,3 unused
//   probe i at RT_LAB_PROBE_HEADER_WORDS + i * RT_LAB_PROBE_WORDS:
//     0 flags (bit 0 geometry, bit 1 variance, bit 2 distribution A + B, bit 3 distribution C)   1,2 screen pixel   3,4 lab pixel   5 linear depth
//     6..9   temporal output (rgb mean in scene radiance units, history length)
//     10..13 filtered output (post spatial; rgb, effective sample count)
//     14..17 this frame's raw mean (rgb, rays shot this frame)
//     18..21 resolved output (rgb)
//     22..24 per sample variance rgb (scene radiance units squared)
//     25 exposure
//     26..29 distribution A (body mean, variance, 3rd central moment, count)    (y = ln(1 + scaled luminance), see rt_lab_temporal.hlsl)
//     30..33 distribution B (tail fraction, tail mean, tail variance, tail count)
//     34..37 distribution C (dark tail fraction, mean, variance, count)
//     38..41 weights of the FIRST spatial pass with this probe as the center and the other probe as the tap (without the kernel, ray length and SH
//            guides): geometry (plane x normal), variance guide, distribution guide, product    42 flags (bit 0 valid, bit 1 variance guide on, bit 2 distribution guide on)
#define RT_LAB_PROBE_COUNT 2
#define RT_LAB_PROBE_SLOTS 8
#define RT_LAB_PROBE_HEADER_WORDS 4
#define RT_LAB_PROBE_WORDS 48
#define RT_LAB_PROBE_SLOT_WORDS (RT_LAB_PROBE_HEADER_WORDS + RT_LAB_PROBE_COUNT * RT_LAB_PROBE_WORDS)

#define RT_LAB_PROBE_FLAG_GEOMETRY 1u
#define RT_LAB_PROBE_FLAG_VARIANCE 2u
#define RT_LAB_PROBE_FLAG_DIST_AB 4u
#define RT_LAB_PROBE_FLAG_DIST_C 8u
#define RT_LAB_PROBE_FLAG_SPATIAL_RAN 16u // push.bound only: the filtered output holds the spatial effective count (/ RT_LAB_SPATIAL_COUNT_SCALE)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabProbeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_temporal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_filtered)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_raw)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, resolved)
// Optional (null when off, push.bound says which are bound).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_variance)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, dist_a)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, dist_b)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, dist_c)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), probe) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLabProbePush
{
    daxa_BufferPtr(RtLabProbeH::AttachmentShaderBlob) attach;
    daxa_u32vec2 screen_pixel[RT_LAB_PROBE_COUNT];
    daxa_u32vec2 lab_pixel[RT_LAB_PROBE_COUNT];
    daxa_u32 slot;
    daxa_u32 stamp;
    daxa_u32 active_mask; // bit i: probe i has a pixel
    daxa_u32 bound;       // RT_LAB_PROBE_FLAG_VARIANCE / _DIST_AB / _DIST_C: that image is bound, RT_LAB_PROBE_FLAG_SPATIAL_RAN
};
