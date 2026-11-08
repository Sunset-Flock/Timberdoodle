#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"
#include "../../../shader_shared/geometry.inl"
#include "../../../shader_shared/scene.inl"
#include "../../../shader_shared/raytracing.inl"
#include "../../../shader_shared/rtgi.inl"

#include "rt_lab_rays.inl"

#define RT_LAB_RAY_T_MAX 100000000000.0f // lab ray t max

// Attachments of the RTGI shading core (rtgi_shade.hlsl). They MUST keep the names of RtgiShadeH: they fill
// RTGI_SHADE_CONTEXT. Shared by the trace and the temporal gradient re-trace, which must shade identically.
#define RT_LAB_SHADE_ATTACHMENTS                                                                    \
    DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)              \
    DAXA_TH_IMAGE_ID(SAMPLE, REGULAR_2D, sky)                                                       \
    DAXA_TH_IMAGE_ID(SAMPLE, REGULAR_2D, sky_transmittance)                                         \
    DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_u32vec4>, light_mask_volume)         \
    DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_f32vec4>, pgi_color)                 \
    DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_f32vec2>, pgi_visibility)            \
    DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_f32vec4>, pgi_info)                  \
    DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayIndex<daxa_u32>, pgi_requests)            \
    DAXA_TH_TLAS_PTR(READ, tlas)                                                                    \
    DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)                               \
    DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)                       \
    DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)                         \
    DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32>, vsm_memory_block)                      \
    DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)

// One cosine distributed diffuse ray per ray list entry (rt_lab_rays.inl: the distribute pass decides how many rays
// each pixel gets), traced with an inline ray query and shaded with the RTGI shading core (rtgi_shade.hlsl: PGI radiance
// on hit, sky on miss). One thread per entry, entries past the frame's budget exit.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabTraceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)                      // linear depth, 0 = sky
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)                // octahedral 32
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ray_list)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(RtLabRay), rays)
RT_LAB_SHADE_ATTACHMENTS
// Ray length guide (RtLabSettings::ray_length_guide_mode, null when off): hit distance t per ray list entry
// (miss = RT_LAB_RAY_T_MAX).
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_f32), ray_lengths)
DAXA_DECL_TASK_HEAD_END

struct RtLabTracePush
{
    daxa_BufferPtr(RtLabTraceH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // lab resolution
    daxa_u32 ray_capacity; // ray list entries
    daxa_u32 ray_lengths_bound; // ray_lengths is bound
};

// Pixel telemetry reference rays (rt_lab_telemetry.hpp): RT_LAB_TELEMETRY_REFERENCE_RAYS cosine rays from ONE lab pixel, traced + shaded exactly like
// the lab trace, all in one frame: ray i uses the frame counter advanced by i (its own noise sample). Never part of the denoiser, only a reference
// for the statistics. Host visible `reference` words: header (RT_LAB_TELEMETRY_REFERENCE_HEADER_WORDS: 0 stamp, 1 geometry flag), then per ray
// RT_LAB_TELEMETRY_REFERENCE_RAY_WORDS floats: radiance rgb (scene units), hit distance (RT_LAB_RAY_T_MAX = miss), direction xyz, 0.
#define RT_LAB_TELEMETRY_REFERENCE_RAYS 4096
#define RT_LAB_TELEMETRY_REFERENCE_THREADS 64
#define RT_LAB_TELEMETRY_REFERENCE_HEADER_WORDS 8
#define RT_LAB_TELEMETRY_REFERENCE_RAY_WORDS 8
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabTelemetryReferenceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
RT_LAB_SHADE_ATTACHMENTS
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), reference)
DAXA_DECL_TASK_HEAD_END

struct RtLabTelemetryReferencePush
{
    daxa_BufferPtr(RtLabTelemetryReferenceH::AttachmentShaderBlob) attach;
    daxa_u32vec2 lab_pixel;
    daxa_u32vec2 size; // lab resolution
    daxa_u32 stamp;
};

// Temporal gradient re-trace (A-SVGF style, rt_lab_rays.inl), one thread per gradient cell record of LAST frame: the
// record's surface point is reprojected to the nearest current lab pixel; if that pixel still shows the same surface
// (plane + normal test), the identical ray (origin, direction, t min) is traced and shaded again and the luminance
// change + max(old, new) (exposed) are written, packed f16, into the current cell that
// pixel falls in (gradient_samples, cleared every frame; colliding cells keep one sample).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGradientTraceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)                 // full res NDC depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)        // half res linear depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, gradient_record_origin)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, gradient_record_surface)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, gradient_record_data)
// Per cell (last writer wins), cleared every frame: exposed old luminance, exposed new luminance, ray direction as
// octahedral coordinates (.z = u + 1, .w = v, .z = 0 means no sample) (rt_lab_gradient_oct.inl).
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_oct_sample)
// World position of the sample's surface point (current frame) + 1 in .w (cleared every frame, 0 = none).
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_oct_position)
RT_LAB_SHADE_ATTACHMENTS
DAXA_DECL_TASK_HEAD_END

struct RtLabGradientTracePush
{
    daxa_BufferPtr(RtLabGradientTraceH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // records (gradient cells, or lab pixels with per_pixel)
    daxa_u32vec2 lab_size; // lab resolution
};
