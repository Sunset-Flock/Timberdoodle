#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"

struct PGIIndirections
{
    DispatchIndirectStruct probe_update_dispatch;
    DispatchIndirectStruct probe_trace_dispatch;
    DispatchIndirectStruct probe_radiance_update_dispatch;
    DispatchIndirectStruct probe_visibility_update_dispatch;
    daxa_u32 indirect_probes;
};

DAXA_DECL_TASK_HEAD_BEGIN(PGIDrawDebugProbesH)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(GRAPHICS_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_u32>, probe_requests)
DAXA_TH_TLAS_ID(GRAPHICS_SHADER_READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct PGIDrawDebugProbesPush
{
    PGIDrawDebugProbesH::AttachmentShaderBlob attach;
    daxa_f32vec3* probe_mesh_positions;
};

#define PGI_UPDATE_WG_XY 8
#define PGI_UPDATE_WG_Z 1

DAXA_DECL_TASK_HEAD_BEGIN(PGIUpdateProbeTexelsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_DECL_TASK_HEAD_END

struct PGIUpdateProbeTexelsPush
{
    PGIUpdateProbeTexelsH::AttachmentShaderBlob attach;
};

DAXA_DECL_TASK_HEAD_BEGIN(PGIUpdateProbesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info_prev)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, requests)
DAXA_DECL_TASK_HEAD_END

struct PGIUpdateProbesPush
{
    PGIUpdateProbesH::AttachmentShaderBlob attach;
};

DAXA_DECL_TASK_HEAD_BEGIN(PGITraceProbeLightingH)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(RAY_TRACING_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(RAY_TRACING_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(RAY_TRACING_SHADER_SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(RAY_TRACING_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, probe_requests)
DAXA_TH_IMAGE_ID(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_ID(RAY_TRACING_SHADER_SAMPLED, REGULAR_2D, sky)
DAXA_TH_TLAS_ID(RAY_TRACING_SHADER_READ, tlas)
DAXA_TH_IMAGE_TYPED(RAY_TRACING_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_BUFFER_PTR(RAY_TRACING_SHADER_READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_DECL_TASK_HEAD_END

struct PGITraceProbeLightingPush
{
    PGITraceProbeLightingH::AttachmentShaderBlob attach;
};

#define PGI_PRE_UPDATE_XYZ 4

DAXA_DECL_TASK_HEAD_BEGIN(PGIPreUpdateProbesH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, requests)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(PGIIndirections), probe_indirections)
DAXA_DECL_TASK_HEAD_END

struct PGIPreUpdateProbesPush
{
    PGIPreUpdateProbesH::AttachmentShaderBlob attach;
    daxa_u32* workgroups_finished;
    daxa_u32 total_workgroups;
};