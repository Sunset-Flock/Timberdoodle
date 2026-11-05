#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#define RT_LAB_RESOLVE_X 8
#define RT_LAB_RESOLVE_Y 8

// Full res output (rt_lab_resolve.hlsl): divides out RTGI_RADIANCE_SCALE; at half res bilateral upsamples first.
// The split difference (split history on, else null) is upsampled with the same weights (stays RTGI_RADIANCE_SCALE scaled)
// and drives the live post-spatial RT_LAB_SPLIT_ERROR / RELATIVE_ERROR debug views.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabResolveH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_diffuse)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, resolved)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, lab_split_difference)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, split_difference_resolved)
DAXA_DECL_TASK_HEAD_END

struct RtLabResolvePush
{
    daxa_BufferPtr(RtLabResolveH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 split_enabled; // lab_split_difference / split_difference_resolved are bound
    daxa_u32 clamp_input_count; // no spatial pass ran: clamp the sample count (sample_count_input_clamp) here
};
