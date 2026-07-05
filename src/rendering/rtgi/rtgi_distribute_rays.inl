#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_DISTRIBUTE_RAYS_X 8
#define RTGI_DISTRIBUTE_RAYS_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiDistributeRaysH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RtgiRayCounters), ray_counters)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, rtgi_sample_count)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RtgiRayEntry), ray_list)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, pixel_ray_alloc)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, ray_count_image)
DAXA_DECL_TASK_HEAD_END

struct RtgiDistributeRaysPush
{
    daxa_BufferPtr(RtgiDistributeRaysH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};
