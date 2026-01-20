#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_DENOISE_DIFFUSE_X 8
#define RTGI_DENOISE_DIFFUSE_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiReprojectDiffuseH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DIndex<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DIndex<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32>, rtgi_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32>, rtgi_samplecnt_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_u32>, rtgi_face_normal_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, rtgi_samplecnt)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_DECL_TASK_HEAD_END

struct RtgiReprojectDiffusePush
{
    RtgiReprojectDiffuseH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};

#define RTGI_DIFFUSE_TEMPORAL_STABILIZATION_X 8
#define RTGI_DIFFUSE_TEMPORAL_STABILIZATION_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiDiffuseTemporalStabilizationH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_diffuse_blurred)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec2>, rtgi_diffuse2_blurred)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_diffuse_reprojected)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec2>, rtgi_diffuse2_reprojected)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_f32>, rtgi_samplecnt)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_stable)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_stable)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_accumulated)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_face_normals)
DAXA_DECL_TASK_HEAD_END

struct RtgiDiffuseTemporalStabilizationPush
{
    RtgiDiffuseTemporalStabilizationH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};