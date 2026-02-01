#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/scene.inl"

#define RTGI_PRE_BLUR_FLATTEN_X 16
#define RTGI_PRE_BLUR_FLATTEN_Y 16

#define RTGI_PRE_BLUR_PREPARE_X 16
#define RTGI_PRE_BLUR_PREPARE_Y 16

#define RTGI_PRE_BLUR_APPLY_X 16
#define RTGI_PRE_BLUR_APPLY_Y 16

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiPreBlurFlattenH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_diffuse_raw)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec2>, rtgi_diffuse2_raw)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, rtgi_flattened_diffuse)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, rtgi_flattened_diffuse2)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_DECL_TASK_HEAD_END

struct RtgiPreBlurFlattenPush
{
    RtgiPreBlurFlattenH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiPreblurPrepareH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_diffuse_raw)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec2>, rtgi_diffuse2_raw)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, rtgi_samplecnt)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_normals)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, rtgi_reconstructed_diffuse_history, 5)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, rtgi_reconstructed_diffuse2_history, 5)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_DECL_TASK_HEAD_END

struct RtgiPreblurPreparePush
{
    RtgiPreblurPrepareH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiPreBlurApply)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_reconstructed_diffuse_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_reconstructed_diffuse2_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, rtgi_samplecnt)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_filtered)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_filtered)
DAXA_DECL_TASK_HEAD_END

struct RtgiPreBlurApplyPush
{
    RtgiPreBlurApply::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};