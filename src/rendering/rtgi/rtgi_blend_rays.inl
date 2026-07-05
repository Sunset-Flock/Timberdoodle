#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_BLEND_RAYS_X 8
#define RTGI_BLEND_RAYS_Y 8

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiBlendRaysH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
// Per-pixel (ray_offset, ray_count) written by the allocate pass.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, pixel_ray_alloc)
// Per-ray radiance results written by the ray-list trace pass.
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiRayResult), ray_result)
// Depth and normals for half-res geometry.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)
// Output consumed by the pre-filter. .rgb = mean log(rgb) geometric mean, .a = mean ray shortness [0,1].
// (diffuse / diffuse2 are no longer produced here — the pre-filter re-blends the rays itself.)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, perceptual_rgb_shortness)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, ray_count_image)
DAXA_DECL_TASK_HEAD_END

struct RtgiBlendRaysPush
{
    daxa_BufferPtr(RtgiBlendRaysH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
};
