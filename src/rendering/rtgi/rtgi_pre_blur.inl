#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_PRE_BLUR_X 8
#define RTGI_PRE_BLUR_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiPreBlurH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, rtgi_diffuse_before)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, rtgi_diffuse2_before)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_diffuse_blurred)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, rtgi_diffuse2_blurred)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, firefly_factor_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, perceptual_radiance_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, ao_guide_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, footprint_quality_image)
// Rays shot per pixel this frame; optionally folded into the blur sample weights (well-sampled pixels
// contribute more), gated by rtgi_settings.pre_blur_ray_count_sample_weighting.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, ray_count_image)
DAXA_DECL_TASK_HEAD_END

struct RtgiPreBlurPush
{
    RtgiPreBlurH::AttachmentShaderBlob attach;
    // Image-attachment count is now EVEN (12) -> blob content is 0-mod-8, so C++ and Slang agree without a
    // padding member. If you add/remove an image attachment and the count becomes ODD again, reinstate a
    // `daxa_u64 _pad_attach;` here to force Slang to match C++'s 4-byte tail padding. See RtgiPreFilterPush.
    daxa_u32vec2 size;
    daxa_u32 iteration;
};