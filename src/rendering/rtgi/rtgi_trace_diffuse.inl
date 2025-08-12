#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/raytracing.inl"

DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(RtgiTraceDiffuseH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_raw)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_raw)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_u32vec4>, light_mask_volume)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec4>, pgi_irradiance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec2>, pgi_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec4>, pgi_info)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayIndex<daxa_u32>, pgi_requests)
DAXA_TH_TLAS_ID(READ, tlas)
// VSM:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, vsm_memory_block)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_DECL_TASK_HEAD_END

struct RtgiTraceDiffusePush
{
    daxa_BufferPtr(RtgiTraceDiffuseH::AttachmentShaderBlob) attach;
    daxa::b32 debug_primary_trace;
};