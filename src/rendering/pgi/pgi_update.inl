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
    DispatchIndirectStruct probe_shade_rays_dispatch;
    DrawIndexedIndirectStruct probe_debug_draw_dispatch;
    daxa_u32 detailed_probe_count;
    daxa_u32 probe_update_count;
};

DAXA_DECL_RASTER_TASK_HEAD_BEGIN(PGIDrawDebugProbesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32), luminance_average)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image)
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_u32>, probe_requests)
DAXA_TH_TLAS_ID(READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct PGIDrawDebugProbesPush
{
    PGIDrawDebugProbesH::AttachmentShaderBlob attach;
    daxa_f32vec3* probe_mesh_positions;
};

#define PGI_UPDATE_WG_XY 8
#define PGI_UPDATE_WG_Z 1

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIUpdateProbeTexelsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_DECL_TASK_HEAD_END

struct PGIUpdateProbeTexelsPush
{
    PGIUpdateProbeTexelsH::AttachmentShaderBlob attach;
    bool update_radiance;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIUpdateProbesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info_copy)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, requests)
DAXA_DECL_TASK_HEAD_END

struct PGIUpdateProbesPush
{
    PGIUpdateProbesH::AttachmentShaderBlob attach;
};

DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(PGITraceProbeLightingH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayIndex<daxa_u32>, probe_requests)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky)
DAXA_TH_TLAS_ID(READ, tlas)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayIndex<daxa_f32vec4>, trace_result)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_DECL_TASK_HEAD_END

struct PGITraceProbeLightingPush
{
    PGITraceProbeLightingH::AttachmentShaderBlob attach;
    GPUScene scene;
};

#define PGI_PRE_UPDATE_XYZ 4

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIPreUpdateProbesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info_copy)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, requests)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(PGIIndirections), probe_indirections)
DAXA_DECL_TASK_HEAD_END

struct PGIPreUpdateProbesPush
{
    PGIPreUpdateProbesH::AttachmentShaderBlob attach;
    daxa_u32* workgroups_finished;
    daxa_u32 total_workgroups;
};

#define PGI_EVAL_SCREEN_IRRADIANCE_XY 8
#define PGI_EVAL_SCREEN_IRRADIANCE_Z 1

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIEvalScreenIrradianceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, main_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, main_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, main_cam_detail_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec4>, probe_radiance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, probe_requests)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, irradiance_depth)
// DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_u32>, normals)
DAXA_DECL_TASK_HEAD_END

struct PGIEvalScreenIrradiancePush
{
    PGIEvalScreenIrradianceH::AttachmentShaderBlob attach;
    daxa_u32vec2 render_target_size;
    daxa_u32vec2 irradiance_image_size;
};

#define PGI_UPSCALE_SCREEN_IRRADIANCE_XY 8
#define PGI_UPSCALE_SCREEN_IRRADIANCE_Z 1

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIUpscaleScreenIrradianceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, main_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, main_cam_detail_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, half_irradiance_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, half_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, full_res_pgi_irradiance)
DAXA_DECL_TASK_HEAD_END

struct PGIUpscaleScreenIrradiancePush
{
    PGIUpscaleScreenIrradianceH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};