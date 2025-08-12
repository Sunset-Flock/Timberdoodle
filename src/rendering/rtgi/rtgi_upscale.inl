#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_UPSCALE_DIFFUSE_X 8
#define RTGI_UPSCALE_DIFFUSE_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiUpscaleDiffuseH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_diffuse_half_res)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec2>, rtgi_diffuse2_half_res)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_camera_detail_normal_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_full_res)
DAXA_DECL_TASK_HEAD_END

struct RtgiUpscaleDiffusePush
{
    RtgiUpscaleDiffuseH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};