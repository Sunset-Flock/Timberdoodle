#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_POST_BLUR_X 16
#define RTGI_POST_BLUR_Y 4

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiPostBlurH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, rtgi_sample_count)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, rtgi_diffuse_before)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, rtgi_diffuse2_before)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_blurred)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_blurred)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, perceptual_radiance_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, ao_guide_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, footprint_quality_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, temporal_perceptual_radiance)
DAXA_DECL_TASK_HEAD_END

struct RtgiPostBlurPush
{
    RtgiPostBlurH::AttachmentShaderBlob attach;
    // Odd image-attachment count -> blob content is 4-mod-8. u64 pad aligns C++ (which pads the blob's
    // sizeof to 8) and Slang (which does not) to the same offset. See RtgiPreFilterPush.
    daxa_u64 _pad_attach;
    daxa_u32vec2 size;
    daxa_b32 pass;
};

struct RtgiAtrousPostBlurPush
{
    RtgiPostBlurH::AttachmentShaderBlob attach;
    // Odd image-attachment count -> blob content is 4-mod-8. See RtgiPostBlurPush / RtgiPreFilterPush.
    daxa_u64 _pad_attach;
    daxa_u32vec2 size;
    daxa_i32 step_size;
};