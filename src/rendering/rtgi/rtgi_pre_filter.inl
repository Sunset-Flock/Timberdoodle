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
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32vec4>, diffuse_raw)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32vec2>, diffuse2_raw)
// Per-pixel: .rgb = geometric mean of the pixel's rays in log space (mean log(rgb)); .a = mean ray
// shortness [0,1]. Perceptual radiance is inferred from .rgb (perceptual_radiance_from_rgb) — used for all
// geometric-mean / firefly-ceiling calculations, and the shortness feeds the ambient-occlusion guide.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32vec4>, perceptual_rgb_shortness)
// Rays shot per pixel this frame; the firefly ceiling is divided by this (well-sampled pixels need less headroom).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, ray_count_image)
// Repacked ray-list data: per-pixel ray-list offset and the per-ray radiance results, so the pre-filter
// can firefly-clamp each ray then re-blend them into the pixel's filtered diffuse.
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
// Footprint quality [0,1], kept SEPARATE from the ambient occlusion guide: multiplying the two before
// temporal accumulation caused streaking, so footprint quality is stored on its own (temporally stable,
// not accumulated) and multiplied into the guide only at pre-blur / post-blur consumption time.
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, footprint_quality_image)
DAXA_DECL_TASK_HEAD_END

struct RtgiPreFilterPush
{
    RtgiPreFilterH::AttachmentShaderBlob attach;
    // Image-attachment count is now EVEN (16) -> blob content is 0-mod-8, so C++ and Slang agree without a
    // padding member. If you add/remove an image attachment and the count becomes ODD again, reinstate a
    // `daxa_u64 _pad_attach;` here to force Slang to match C++'s 4-byte tail padding.
    daxa_u32vec2 size;
};