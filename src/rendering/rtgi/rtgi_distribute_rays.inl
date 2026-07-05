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
// Global demand (total_extra_rays) and write cursor (ray_list_count) for the ray list.
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RtgiRayDemand), ray_demand)
// Reprojected sample count per pixel, used to derive per-pixel desired ray count.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, rtgi_sample_count)
// Output: flat ray list filled atomically per workgroup.
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RtgiRayEntry), ray_list)
// Output: per-pixel ray_offset into the ray list (ray count lives in ray_count_image, not duplicated here).
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, pixel_ray_alloc)
// Output: rays allocated to each pixel this frame (R8_UINT). Written here so the blend pass and the
// pre-filter/temporal passes all read the count from one place instead of a redundant alloc channel.
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, ray_count_image)
DAXA_DECL_TASK_HEAD_END

struct RtgiDistributeRaysPush
{
    daxa_BufferPtr(RtgiDistributeRaysH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};
