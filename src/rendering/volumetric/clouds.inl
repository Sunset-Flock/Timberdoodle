#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RAYMARCH_CLOUDS_DISPATCH_X 8
#define RAYMARCH_CLOUDS_DISPATCH_Y 8

// Debug raymarch just marches a single ray and does not write anything to the raymarched result.
// I leave the same head for both - this is suboptimal, but since this is a debug utility I deep it fine.
// Especially given that I dispatch a single thread etc etc...
#define RAYMARCH_CLOUDS_DEBUG_DISPATCH_X 1
#define RAYMARCH_CLOUDS_DEBUG_DISPATCH_Y 1

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RaymarchCloudsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture3DId<daxa_f32vec2>, cloud_data_field)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture3DId<daxa_f32vec4>, cloud_detail_noise)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, transmittance)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, depth)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::TextureCubeId<daxa_f32vec4>, sky_ibl)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, clouds_raymarched_result)
DAXA_DECL_TASK_HEAD_END

struct RaymarchCloudsPush
{
    daxa_BufferPtr(RaymarchCloudsH::AttachmentShaderBlob) attach;
    daxa_u32vec2 clouds_resolution;
};

#define COMPOSE_CLOUDS_DISPATCH_X 8
#define COMPOSE_CLOUDS_DISPATCH_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ComposeCloudsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32vec4>, clouds_raymarched_result)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_f32vec3>, color_image)
DAXA_DECL_TASK_HEAD_END

struct ComposeCloudsPush
{
    daxa_BufferPtr(ComposeCloudsH::AttachmentShaderBlob) attach;
    daxa_u32vec2 main_screen_resolution;
};