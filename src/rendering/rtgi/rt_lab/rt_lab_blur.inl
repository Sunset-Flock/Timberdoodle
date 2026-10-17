#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#define RT_LAB_BLUR_X 8
#define RT_LAB_BLUR_Y 8

// One pass of the separable bilateral gaussian (rt_lab_blur.hlsl), geometric edge stopping only.
// Alpha = effective sample count: the temporal history length (clamped to sample_count_input_clamp in the first pass)
// accumulated as sum(weight / center weight * count) without normalization. The split difference
// (split history on, else null) is blurred with the same weights: the weights only depend on geometry, so
// blur(A) - blur(B) == blur(A - B) and |blurred difference| stays the error estimate of the blurred diffuse.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabBlurH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dst)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, split_src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, split_dst)
// Per sample rgb variance from the temporal pass (variance guide, fine filter only; null without temporal).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_variance)
DAXA_DECL_TASK_HEAD_END

struct RtLabBlurPush
{
    daxa_BufferPtr(RtLabBlurH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_i32vec2 direction; // (1,0) horizontal, (0,1) vertical
    daxa_u32 split_enabled;  // split_src / split_dst are bound
    daxa_i32 step;           // a-trous only: tap spacing in pixels (1, 2, 4, 8)
    daxa_u32 clamp_input_count; // first spatial pass: clamp the input sample counts (sample_count_input_clamp)
    daxa_u32 variance_bound;    // lab_variance is bound (temporal on)
};
