#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../../shader_shared/shared.inl"
#include "../../../../shader_shared/globals.inl"

#define RT_LAB_NEURAL_X 8
#define RT_LAB_NEURAL_Y 8

// Neural latent pass (rt_lab_neural.hlsl entry_rt_lab_neural_latent): one thread per lab pixel runs the shared network
// (rt_lab_neural_core.slang) on the pixel's temporal output + its reprojected previous latent and writes this frame's
// latent: 8 f16 packed in an R32G32B32A32_UINT texel (persistent double buffer). Runs between the temporal pass and the
// first spatial pass; the neural a-trous passes then weight every tap with exp(-|f_c - f_t|^2).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabNeuralLatentH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
// The temporal pass output (rgb RTGI_RADIANCE_SCALE units, .a history length), this frame's raw samples (rgb mean, .a rays)
// and the accumulated per sample variance (rgb).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_raw_diffuse)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_variance)
// Ray length guide (null when off, push.ray_length_bound): .x ao guide (< 0 = none), .y mean shortness.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, lab_ray_length)
// Lite feature set inputs, each null when unavailable (push flags): the full res detail normal (packed octahedral, only when the lab runs
// at full res) and the accumulated per pixel capped normalized ray length (RtLabNeuralRayStatsH, -1 = never hit by a ray).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, detail_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, ray_capped)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, latent_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, latent)
// Network outputs [8, 16) (weighting function gate logits, sharpness) as 8 packed f16, per frame (not recurrent).
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, affinity)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32), weights)
// The same weights as f16 (row major, same offsets) for the cooperative vector entry (Slang CoopVec wants a ByteAddressBuffer, not a pointer).
DAXA_TH_BUFFER_ID(READ, weights_f16)
DAXA_DECL_TASK_HEAD_END

struct RtLabNeuralLatentPush
{
    daxa_BufferPtr(RtLabNeuralLatentH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 ray_length_bound;
    daxa_u32 reset; // the history was reset this frame: previous latents are meaningless
    daxa_u32 detail_normal_bound;
    daxa_u32 ray_capped_bound;
};

// Ray statistics pass (entry_rt_lab_neural_ray_stats, lite feature set only): the per pixel capped normalized ray length the network was trained
// with. This frame's rays of a pixel (ray_alloc = offset / count into ray_lengths) give the mean of min(t, 32 pixel widths) / (32 pixel widths);
// like the lab's temporal pass accumulates colors it is blended into the reprojected history of the pixel with n / history length (n = rays of the
// frame, history length = accumulated.a). Pixels without a ray this frame keep their history, pixels never hit by a ray hold -1.
// Runs between the temporal pass and the latent pass (train/dataset.py accumulate_ray_stats is the offline twin).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabNeuralRayStatsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_raw_diffuse)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32), ray_lengths)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, ray_capped_history)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, ray_capped)
DAXA_DECL_TASK_HEAD_END

struct RtLabNeuralRayStatsPush
{
    daxa_BufferPtr(RtLabNeuralRayStatsH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 reset; // the history was reset this frame: no previous values
};

// Neural a-trous pass (entry_rt_lab_neural_atrous): the lab's fine a-trous (rt_lab_blur.hlsl entry_rt_lab_atrous) with the
// shared core's pass: geometric edge stopping x the learned affinity of the latents (no lab guide).
// Same input / output convention (alpha = effective sample count, split difference carried with the same weights).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabNeuralAtrousH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, dst)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, split_src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, split_dst)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_variance)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, latent)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, affinity)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, history_feedback)
DAXA_DECL_TASK_HEAD_END

struct RtLabNeuralAtrousPush
{
    daxa_BufferPtr(RtLabNeuralAtrousH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_i32 step;               // tap spacing in pixels (1, 2, 4, ...)
    daxa_u32 split_enabled;      // split_src / split_dst are bound
    daxa_u32 clamp_input_count;  // first pass: the input is the temporal history (plain history length, sample_count_input_clamp)
    daxa_u32 history_feedback;   // history_feedback is bound (first a-trous pass)
    daxa_f32 target_samples;     // ring budget of the pass (effective samples)
};

// Training data dump (RtLabNeuralDump in rt_lab_neural.hpp): copies a square crop of the lab data into a host visible
// buffer, one block per frame. The geometry block (written once) is position / normal / depth / rcp pixel width per
// pixel (f32) + [2] detail normal.xyz, albedo.rgb (packed f16, 0 when unavailable), a frame block is a header uint4 (exposure bits, frame
// index) + 2 uint4 of packed f16 per pixel:
//   [0] acc.rgb, acc.a (history length), raw.rgb, raw.a (rays)
//   [1] variance.rgb, ray ao, ray shortness, this frame's mean log2(1 + hit distance) of the pixel's rays, mean capped normalized
//       hit distance (min(t, 32 pixel widths) / (32 pixel widths)), ray count (-1 / -1 / 0 = no rays or no ray data)
// Version 3 adds two uint4 per pixel (4 uint4 per pixel and frame):
//   [2] SH-Y xyz, SH-Y w (= mean Y), Co, Cg (packed f16), 2 f16 spare   [3] mean of Y^2, Co^2, Cg^2 and a spare (f32)
// train/dataset.py reads it.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabNeuralDumpH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, accumulated)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_raw_diffuse)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_variance)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, lab_ray_length)
// Dump v3 (SH feature set data), each null when unavailable (push.sh_bound): the temporal SH-Y (xyz = direction x Y, w = Y), CoCg (.xy) + history length (.z)
// and the squared moments of Y, Co, Cg (.xyz).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_sh_y)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_cocg)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, lab_sq)
// Dump v2 (raw 3x3 feature set data), each null when unavailable (push flags): the full res detail normal (packed octahedral, only
// when the lab runs at full res), the half res albedo (RGBA8, sampled at pixel >> push.albedo_shift) and this frame's rays of every
// pixel (ray_alloc = offset / count into ray_lengths) for the per pixel ray length statistics.
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, detail_normal)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, half_res_albedo)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32), ray_lengths)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32vec4), dump) // host visible
DAXA_DECL_TASK_HEAD_END

struct RtLabNeuralDumpPush
{
    daxa_BufferPtr(RtLabNeuralDumpH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;        // lab size
    daxa_u32vec2 crop_origin;
    daxa_u32 crop_size;       // square crop edge
    daxa_u32 block;           // 0 = geometry block, 1 + k = frame block k
    daxa_u32 ray_length_bound;
    daxa_u32 frame_index;
    daxa_u32 detail_normal_bound;
    daxa_u32 albedo_bound;
    daxa_u32 albedo_shift;       // lab pixel >> albedo_shift = albedo pixel (0 half res lab, 1 full res lab)
    daxa_u32 ray_stats_bound;    // ray_alloc / ray_lengths are bound
    daxa_u32 sh_bound;           // lab_sh_y / lab_cocg / lab_sq are bound (dump v3)
};
