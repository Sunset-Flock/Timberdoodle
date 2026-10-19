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
// Gradient aware variance guide (variance_guide_gradient, push.gradient_mode): the local rgb gradient per screen axis
// (.rgb slope per pixel, .a its variance), fitted in the first fine pass, filtered with the color's weights by the next
// ones (rt_lab_blur.hlsl). Null when off; src null in the first pass.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, gradient_x_src)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, gradient_y_src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_x_dst)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_y_dst)
// Ray length guide (RtLabSettings::ray_length_guide_mode, push.ray_length_bound): the temporal pass's accumulated
// (.x AO guide 1 - sqrt(mean RTGI ray shortness), .y mean shortness); guide radius = .x x R. Null when off.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, lab_ray_length)
// RT_LAB_COLOR_GRADIENT debug view (last fine pass).
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
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
    daxa_u32 gradient_mode;     // RT_LAB_GRADIENT_*
    daxa_u32 last_pass;         // last fine pass: draws RT_LAB_COLOR_GRADIENT
    daxa_u32 firefly_weight;    // first fine pass + firefly_weight_enabled + lab_variance .a holds the raw luminance
    daxa_u32 ray_length_bound;  // lab_ray_length is bound
};
#define RT_LAB_GRADIENT_OFF 0u
#define RT_LAB_GRADIENT_FIT 1u   // first fine pass: fit the gradient from the color, write it (its own guide: no gradient)
#define RT_LAB_GRADIENT_CARRY 2u // later passes: read it, filter it with the color's weights, write it
