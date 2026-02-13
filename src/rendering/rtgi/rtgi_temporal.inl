#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_TEMPORAL_X 8
#define RTGI_TEMPORAL_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiTemporalH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DIndex<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DIndex<daxa_u32>, clocks_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, half_res_sample_count)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32>, half_res_sample_count_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32vec4>, half_res_diffuse_new)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, half_res_diffuse_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32vec4>, half_res_diffuse_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32vec2>, half_res_diffuse2_new)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, half_res_diffuse2_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32vec2>, half_res_diffuse2_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, half_res_statistics_accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_u32>, half_res_statistics_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32>, half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_f32>, half_res_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_u32>, half_res_normal)           // probably best to use face or smooth normal here
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DIndex<daxa_u32>, half_res_normal_history)   // probably best to use face or smooth normal here
DAXA_DECL_TASK_HEAD_END

struct RtgiTemporalPush
{
    RtgiTemporalH::AttachmentShaderBlob attach;
    daxa_u64 padd;
    daxa_u32vec2 size;
};