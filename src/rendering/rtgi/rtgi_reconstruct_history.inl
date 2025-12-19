#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/scene.inl"

#define RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_X 16
#define RTGI_RECONSTRUCT_HISTORY_GEN_MIPS_DIFFUSE_Y 16

#define RTGI_RECONSTRUCT_HISTORY_APPLY_DIFFUSE_X 16
#define RTGI_RECONSTRUCT_HISTORY_APPLY_DIFFUSE_Y 16

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiReconstructHistoryGenMipsDiffuseH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_diffuse_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec2>, rtgi_diffuse2_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, rtgi_samplecnt)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, rtgi_reconstructed_diffuse_history, 5)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, rtgi_reconstructed_diffuse2_history, 5)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_DECL_TASK_HEAD_END

struct RtgiReconstructHistoryGenMipsDiffusePush
{
    RtgiReconstructHistoryGenMipsDiffuseH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiReconstructHistoryApplyDiffuseH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_reconstructed_diffuse_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_reconstructed_diffuse2_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, rtgi_samplecnt)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_accumulated)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_accumulated)
DAXA_DECL_TASK_HEAD_END

struct RtgiReconstructHistoryApplyDiffusePush
{
    RtgiReconstructHistoryApplyDiffuseH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};