#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_PRE_BLUR_PREPARE_X 8
#define RTGI_PRE_BLUR_PREPARE_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiPreFilterH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32vec4>, perceptual_rgb_shortness)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, ray_count_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiRayResult), ray_result)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, view_cam_half_res_albedo)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, pre_filtered_diffuse_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, pre_filtered_diffuse2_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, firefly_factor_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, perceptual_radiance_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, ao_guide_image)
DAXA_DECL_TASK_HEAD_END

struct RtgiPreFilterPush
{
    RtgiPreFilterH::AttachmentShaderBlob attach;
    daxa_u64 _pad_attach;
    daxa_u32vec2 size;
};