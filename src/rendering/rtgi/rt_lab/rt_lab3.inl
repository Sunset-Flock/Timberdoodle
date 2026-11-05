#pragma once

// MUST come before the first daxa include (daxa.inl is pragma once): enables RaytracingAccelerationStructure::get.
#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"
#include "rt_lab_trace.inl" // RT_LAB_SHADE_ATTACHMENTS, tlas / mesh instances, RtLabRayCounters
#include "rt_lab3_settings.inl"

DAXA_DECL_BUFFER_PTR(RtLab3Settings)

// RT Lab 3 (rt_lab3.hpp): ReSTIR GI on the RT Lab trace. Per lab pixel ONE reservoir holds one indirect light sample: the hit point
// x_s of a diffuse ray, its normal n_s and the outgoing radiance L_o there (all from the trace, nothing is re-shaded on reuse).
//   [old lab: ray demand (history = last frame's output, .a = reservoir confidence M) -> distribute]
//   trace      one cosine ray per ray list entry, writes hit position + hit normal + radiance (RtLab3Ray)
//   temporal   (per pixel) the pixel's rays are resampled into a reservoir (RIS, target p_hat = luminance(L_o) cos), merged with the
//              reprojected reservoir of last frame (reconnection Jacobian, M capped, optional visibility ray). The result is the
//              history of the next frame.
//   spatial    (per pixel, 0..RT_LAB3_MAX_SPATIAL_PASSES passes, ping pong) merges neighbor reservoirs (random taps in a disc, geometry
//              tests, Jacobian, optional visibility ray). Not fed back into the history.
//   shade      (per pixel) estimate = L_o cos W / pi from the final reservoir (W = unbiased contribution weight)
//   filter     ray filter AFTER the reuse (clamps the estimate against its neighbors, hue preserving)
//   [old lab: resolve (upsamples at half res)]
// Reuse is the biased ReSTIR GI variant (M weighted, no pairwise MIS). No temporal or spatial denoiser follows.
//
// Reservoir images (lab res): A (RGBA32F) sample position xyz + W, B (RGBA32F) radiance rgb (RTGI_RADIANCE_SCALE units) + M, C (R32G32B32A32_UINT)
// .x octahedral sample normal, .y sample age in frames (0 = traced this frame; +1 per temporal reuse, spatial reuse keeps the age of the copied sample),
// .z asfloat RAY HISTORY of the pixel: rays shot at this pixel over time (reprojected like the temporal reuse, capped at the target of the ray redistribution).
// It is NOT the reservoir confidence M (reuse changes M without any ray being shot): the ray redistribution reads it (output .a) as its history.

#define RT_LAB3_PASS_X 8
#define RT_LAB3_PASS_Y 8
#define RT_LAB3_SKY_DISTANCE 10000.0f // sample distance of rays that miss (the sample point is then a far point in the ray direction)

// One traced ray (RT Lab 3 trace).
struct RtLab3Ray
{
    daxa_f32vec3 radiance;  // rgb * RTGI_RADIANCE_SCALE
    daxa_u32 hit_normal;    // octahedral 32, faces the ray; a miss: -direction
    daxa_f32vec3 position;  // hit point (a miss: RT_LAB3_SKY_DISTANCE along the direction)
    daxa_u32 direction;     // octahedral 32
};
DAXA_DECL_BUFFER_PTR(RtLab3Ray)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab3TraceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ray_list)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(RtLab3Ray), rays)
RT_LAB_SHADE_ATTACHMENTS
DAXA_DECL_TASK_HEAD_END

struct RtLab3TracePush
{
    daxa_BufferPtr(RtLab3TraceH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // lab resolution
    daxa_u32 ray_capacity; // ray list entries
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab3TemporalH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab3Ray), rays)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history_a)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history_b)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, history_c)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, out_a)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, out_b)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, out_c)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct RtLab3TemporalPush
{
    daxa_BufferPtr(RtLab3TemporalH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab3Settings) settings;
    daxa_u32vec2 size;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab3SpatialH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, in_a)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, in_b)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, in_c)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, out_a)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, out_b)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, out_c)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct RtLab3SpatialPush
{
    daxa_BufferPtr(RtLab3SpatialH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab3Settings) settings;
    daxa_u32vec2 size;
    daxa_u32 pass_index;
};

// Final reservoir -> radiance estimate (.rgb RTGI_RADIANCE_SCALE units, .a = the confidence M of the TEMPORAL reservoir: it is what the next
// frame's ray demand reads; the spatial passes add counts of other pixels that say nothing about this pixel's history).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab3ShadeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, in_a)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, in_b)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, in_c)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, shaded)
DAXA_DECL_TASK_HEAD_END

struct RtLab3ShadePush
{
    daxa_BufferPtr(RtLab3ShadeH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab3Settings) settings;
    daxa_u32vec2 size;
};

// Ray filter after the reuse.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLab3FilterH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, shaded)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, filtered)
// RT Lab 4 (null otherwise): the filtered estimate again with a per pixel sample weight in .a (RT_LAB3_CELL_WEIGHT_*), the input of the cell grid.
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, cell_input)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLab3Ray), rays) // RT_LAB3_CELL_SOURCE_RAYS
DAXA_DECL_TASK_HEAD_END

struct RtLab3FilterPush
{
    daxa_BufferPtr(RtLab3FilterH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLab3Settings) settings;
    daxa_u32vec2 size;
    daxa_u32 cell_input_bound; // cell_input is written
    daxa_u32 cell_weight_mode; // RT_LAB3_CELL_WEIGHT_* (cell_source RESTIR)
    daxa_u32 cell_source;      // RT_LAB3_CELL_SOURCE_*
};
