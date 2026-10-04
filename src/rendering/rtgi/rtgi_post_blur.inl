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
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, ao_guide_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, temporal_perceptual_radiance)
// Specular channel (.rgb radiance, .a hit distance) + lobe inputs for its weights.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, rtgi_specular_before)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, rtgi_specular_blurred)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, specular_normal_roughness) // pack_normal_roughness
DAXA_DECL_TASK_HEAD_END

struct RtgiPostBlurPush
{
    daxa_BufferPtr(RtgiPostBlurH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_b32 pass;
};

struct RtgiAtrousPostBlurPush
{
    daxa_BufferPtr(RtgiPostBlurH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_i32 step_size;
};