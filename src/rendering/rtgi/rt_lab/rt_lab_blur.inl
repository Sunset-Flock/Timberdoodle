#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

#define RT_LAB_BLUR_X 8
#define RT_LAB_BLUR_Y 8
// Distribution guide weight (rt_lab_blur.hlsl): prior per sample log variance (one pseudo sample) and the epsilon of the denominator.
#define RT_LAB_DISTRIBUTION_PRIOR_VARIANCE 1.0f
#define RT_LAB_DISTRIBUTION_EPSILON 0.02f

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
// Distribution guide (RtLabSettings::distribution_guide_mode, push.distribution_bound): the temporal pass's body / tail state A
// (body log mean, body log variance, perceptual mean, body count; count < 1 = no state, see rt_lab_temporal.inl). Null when off.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_distribution)
// Tail state B (tail fraction p, tail log mean, tail log variance, tail count), bound with the dark tail mode (push.distribution_multi).
// Null otherwise.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_distribution_tail)
// Dark tail state C (push.distribution_multi: mode 3 weighs all three components, B and C are bound then). Null otherwise.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_distribution_low)
// SH guide (RtLabSettings::sh_guide_mode, push.sh_bound): the temporal pass's SH-Y (xyz = direction x Y, w = Y) and CoCg + history
// length (.z). Null when off.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_sh_y)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_cocg)
// SH second moments (RT_LAB_SH_GUIDE_DIRECTIONAL_VARIANCE, push.sh_m2_bound; A and B, see rt_lab_temporal.inl). Null when off.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_sh_m2)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_sh_m2b)
// Guide mean (push.guide_mean_bound, the first a-trous pass that filters the RAW frame): the temporally accumulated statistics step's mean (rgb) and history
// length (.a) of every pixel. The variance guide compares these instead of the raw samples (which are single, noisy rays). Null otherwise.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_guide_mean)
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
    daxa_u32 sh_bound;          // lab_sh_y / lab_cocg are bound
    daxa_u32 sh_m2_bound;       // lab_sh_m2 is bound
    daxa_u32 guide_mean_bound;  // lab_guide_mean is bound (the pass filters the raw frame)
    daxa_u32 distribution_bound; // lab_distribution is bound
    daxa_u32 distribution_multi;      // mode 3: lab_distribution_tail + lab_distribution_low are bound, the tap weight compares all components
};

