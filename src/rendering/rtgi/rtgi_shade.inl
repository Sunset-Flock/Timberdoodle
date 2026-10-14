#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/raytracing.inl"
#include "../../shader_shared/rtgi.inl"

// Shade passes of the split trace / shade (repacked ray-list path): shade the rays the slim split trace
// (ray_gen_split, rtgi_trace.hlsl) recorded, one compute pass per shade list (rtgi_shade.hlsl), each dispatched
// indirectly over its list:
//   entry_shade_sharp:  specular hits, full material evaluation + light shading (sun shadow via ray query, VSM for
//                       point / spot lights).
//   entry_shade_coarse: diffuse hits, PGI radiance cache only.
//   entry_shade_miss:   misses, sky.
// Separate pipelines, so the cheap passes never pay the register footprint of the full shading. All three are recorded
// in ONE task without barriers between them (disjoint ray_result entries), so they overlap; one timer covers them.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiShadeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ | INDIRECT_COMMAND_READ, daxa_BufferPtr(RtgiShadeCounters), shade_counters)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), shade_list)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtgiRayHit), ray_hits)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtgiRayResult), ray_result)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_ID(SAMPLE, REGULAR_2D, sky)
DAXA_TH_IMAGE_ID(SAMPLE, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_u32vec4>, light_mask_volume)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayIndex<daxa_f32vec4>, pgi_color)
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

struct RtgiShadePush
{
    daxa_BufferPtr(RtgiShadeH::AttachmentShaderBlob) attach;
};
