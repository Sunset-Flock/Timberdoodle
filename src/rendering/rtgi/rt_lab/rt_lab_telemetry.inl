#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#include "rt_lab_rays.inl"

// Pixel telemetry (rt_lab_telemetry.hpp): copies the rays + surface detail of ONE lab pixel into a small host visible ring
// buffer, one slot per collected frame. The CPU accumulates the rays over the frames and computes the statistics.
// One workgroup, one thread per ray. Words (u32 / bit cast f32) of a slot:
//   header (RT_LAB_TELEMETRY_HEADER_WORDS):
//     0 stamp (frame stamp, never 0)   1 rays of the pixel this frame   2 rays stored (<= RT_LAB_TELEMETRY_MAX_RAYS)   3 flags (bit 0 = geometry)
//     4,5 screen pixel   6,7 lab pixel
//     8..10 world position   11 linear depth
//     12..14 face normal     15 metalness
//     16..18 detail normal   19 pixel width (world space, lab pixel)
//     20..22 albedo          23 exposure
//     24..26 view direction (camera -> surface)
//   rays (RT_LAB_TELEMETRY_RAY_WORDS each):
//     0..2 incoming radiance (unmodulated)   3 hit distance (RT_LAB_RAY_T_MAX = miss)
//     4..6 modulated radiance = diffuse albedo x radiance x cos(detail normal) / cos(face normal)   7 cos(detail normal), clamped to >= 0
//     8..10 direction        11 cos(face normal)
// The modulation is the importance weight of the diffuse estimate: the rays are cosine distributed around the FACE normal,
// the shaded pixel integrates cos around the DETAIL normal. The mean of the modulated radiance is the pixel's outgoing
// diffuse radiance (before exposure).
#define RT_LAB_TELEMETRY_MAX_RAYS 64
#define RT_LAB_TELEMETRY_THREADS 64
#define RT_LAB_TELEMETRY_SLOTS 8
#define RT_LAB_TELEMETRY_HEADER_WORDS 32
#define RT_LAB_TELEMETRY_RAY_WORDS 12
#define RT_LAB_TELEMETRY_SLOT_WORDS (RT_LAB_TELEMETRY_HEADER_WORDS + RT_LAB_TELEMETRY_MAX_RAYS * RT_LAB_TELEMETRY_RAY_WORDS)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabTelemetryH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, detail_normal)           // full res, sampled at the screen pixel
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_albedo)     // RGBA8 albedo + metalness, sampled at screen pixel >> 1
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32), ray_lengths)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), telemetry) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLabTelemetryPush
{
    daxa_BufferPtr(RtLabTelemetryH::AttachmentShaderBlob) attach;
    daxa_u32vec2 screen_pixel;
    daxa_u32vec2 lab_pixel;
    daxa_u32 slot;
    daxa_u32 stamp;
};
