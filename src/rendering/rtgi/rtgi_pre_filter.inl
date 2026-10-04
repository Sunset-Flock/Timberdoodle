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
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, pre_filtered_diffuse_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, pre_filtered_diffuse2_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, firefly_factor_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, perceptual_radiance_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, ao_guide_image)
// Specular channel: per-ray specular results, per-pixel log-mean + hit distance (firefly ceiling input) and
// the pre-filtered output (.rgb = firefly-clamped mean specular radiance, .a = mean hit distance).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, specular_ray_count_image) // specular rays follow the pixel's diffuse rays in ray_result
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, specular_perceptual_rgb_hit)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, pre_filtered_specular_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_normal_roughness)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiRayCounters), ray_counters) // statistics -> general readback
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, specular_firefly_factor_image) // specular energy_pre / energy_post of the firefly clamp
DAXA_DECL_TASK_HEAD_END

struct RtgiPreFilterPush
{
    daxa_BufferPtr(RtgiPreFilterH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};