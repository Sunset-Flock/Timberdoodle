#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/raytracing.inl"

DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(RtgiTraceDiffuseH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, diffuse_raw)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, diffuse2_raw)
// Per-pixel: .rgb = geometric mean of this pixel's rays in log space (mean log(rgb)); .a = mean ray
// shortness [0,1]. Perceptual radiance is inferred from .rgb (perceptual_radiance_from_rgb) — the geometric mean
// of the *rays* is needed so the pre-filter's firefly ceiling isn't Jensen-biased high.
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, perceptual_rgb_shortness)
// Number of rays this pixel shot this frame. Read by the accumulation pass (to increment the sample
// count) and the pre-filter (to divide the firefly ceiling — well-sampled pixels need less headroom).
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DId<daxa_u32>, ray_count_image)
// Reprojected history sample count from the temporal reproject pass (runs before trace).
// Read-only here to drive the adaptive ray count; the accumulation pass does the increment. <0 == sky.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_u32>, rtgi_sample_count)
// Per-frame ray demand. READ_WRITE because the classic (non-distributed) trace atomically reserves its
// wave's slice of ray_list_count here (the distribute pass does this in the repacked path).
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtgiRayDemand), ray_demand)
// Flat ray list built by the distribute pass: each entry is a (pixel_xy, sample_index) pair.
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiRayEntry), ray_list)
// Per-ray radiance results: written by ray_gen_from_list (repacked) or shade_ray_gen (classic), read by
// the pre-filter (which re-blends + per-ray firefly-clamps them).
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RtgiRayResult), ray_result)
// Per-pixel ray-list offset. Written by the distribute pass (repacked) or shade_ray_gen (classic).
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, pixel_ray_alloc)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32>, view_cam_half_res_depth) 
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_ID(SAMPLE, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(SAMPLE, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_u32vec4>, light_mask_volume)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_f32vec4>, pgi_irradiance)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_f32vec2>, pgi_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_f32vec4>, pgi_info)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayIndex<daxa_u32>, pgi_requests)
DAXA_TH_TLAS_PTR(READ, tlas)
// VSM:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32>, vsm_memory_block)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_DECL_TASK_HEAD_END

struct RtgiTraceDiffusePush
{
    daxa_BufferPtr(RtgiTraceDiffuseH::AttachmentShaderBlob) attach;
    daxa::b32 debug_primary_trace;
};