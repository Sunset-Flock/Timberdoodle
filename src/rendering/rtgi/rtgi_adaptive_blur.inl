#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_ADAPTIVE_BLUR_DIFFUSE_X 8
#define RTGI_ADAPTIVE_BLUR_DIFFUSE_Y 8

#define RTGI_PRE_BLUR_DIFFUSE_X 8
#define RTGI_PRE_BLUR_DIFFUSE_Y 8

#define RTGI_ATROUS_BLUR_DIFFUSE_X 8
#define RTGI_ATROUS_BLUR_DIFFUSE_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiAdaptiveBlurH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, rtgi_diffuse_before)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec2>, rtgi_diffuse2_before)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, rtgi_samplecnt)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_blurred)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_blurred)
DAXA_DECL_TASK_HEAD_END

struct RtgiAdaptiveBlurPush
{
    RtgiAdaptiveBlurH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
    daxa_b32 pass;
};