#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

// Temporal resolve ("integrator") shared by all RT labs (rt_lab_integrator.hpp). TAA / temporal analyzer hybrid on the lab's final per pixel
// radiance: the window variance of THIS frame's input decides how much history is kept (a quiet window = the input is already accurate: trust
// it, small history; a noisy window keeps more history and moves the new value toward the window mean), the reprojected history (bilinear,
// surface tested) is clamped into mean +- gamma sigma of the window. Input .rgb radiance (RTGI_RADIANCE_SCALE units), .a > valid_threshold =
// the pixel has a value. Output .rgb radiance, .a = history length in frames.

#define RT_LAB_INTEGRATOR_X 8
#define RT_LAB_INTEGRATOR_Y 8
#define RT_LAB_INTEGRATOR_MAX_WINDOW_RADIUS 4

struct RtLabIntegratorSettings
{
    daxa_i32 window_radius;      // variance window (2r + 1)^2 on the input of this frame
    daxa_f32 variance_gamma;     // neighborhood box half width (window std devs) the history is clamped into, 0 = no clamp
    daxa_f32 alpha_min;          // blend factor of the new value at the noisiest windows (keeps most history)
    daxa_f32 alpha_max;          // ... at quiet windows (trusts the new value)
    daxa_f32 variance_reference; // window relative std at which alpha reaches alpha_min
    daxa_f32 spatial_strength;   // 0..1: how far a noisy window moves the new value to the window mean
    daxa_i32 max_frames;         // cap of the history length (alpha never goes below 1 / length)
    daxa_f32 plane_tolerance;    // reprojection / window surface test in pixel widths
    daxa_f32 normal_threshold;   // reprojection / window face normal dot threshold
    daxa_i32 reset;              // set by the CPU: ignore the history this frame
};
DAXA_DECL_BUFFER_PTR(RtLabIntegratorSettings)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabIntegratorH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, input)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history)    // last frame's output
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, integrated) // this frame's output
DAXA_DECL_TASK_HEAD_END

struct RtLabIntegratorPush
{
    daxa_BufferPtr(RtLabIntegratorH::AttachmentShaderBlob) attach;
    daxa_BufferPtr(RtLabIntegratorSettings) settings;
    daxa_u32vec2 size;
};
