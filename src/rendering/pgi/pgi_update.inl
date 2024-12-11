#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"

DAXA_DECL_TASK_HEAD_BEGIN(PGIDrawDebugProbesH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec3>, sh_probes)
DAXA_TH_TLAS_ID(GRAPHICS_SHADER_READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct PGIDrawDebugProbesPush
{
    PGIDrawDebugProbesH::AttachmentShaderBlob attach;
    daxa_f32vec3* probe_mesh_positions;
};

#define PGI_UPDATE_WG_XY 8
#define PGI_UPDATE_WG_Z 1

DAXA_DECL_TASK_HEAD_BEGIN(PGIUpdateProbesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec3>, sh_probes)
DAXA_TH_TLAS_ID(COMPUTE_SHADER_READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct PGIUpdateProbesPush
{
    PGIUpdateProbesH::AttachmentShaderBlob attach;
};

DAXA_DECL_TASK_HEAD_BEGIN(PGITraceProbeLightingH)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(RAY_TRACING_SHADER_STORAGE_READ_ONLY, daxa::Texture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_ID(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_ID(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, sky)
DAXA_TH_TLAS_ID(RAY_TRACING_SHADER_READ, tlas)
DAXA_TH_IMAGE_TYPED(RAY_TRACING_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_DECL_TASK_HEAD_END

struct PGITraceProbeLightingPush
{
    PGITraceProbeLightingH::AttachmentShaderBlob attach;
};