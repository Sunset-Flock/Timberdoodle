#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../scene_renderer_context.hpp"

#include "rt_lab_neural_state.hpp"
#include "rt_lab_neural.inl"

///
/// RT Lab neural affinity (rt_lab.md "Neural affinity"): a small recurrent MLP (neural/rt_lab_neural_core.slang, shared with
/// the trainer in neural/train) predicts a latent feature vector per lab pixel from the temporal output; the lab's a-trous
/// passes then weight every tap with exp(-|f_c - f_t|^2) on top of the geometric tests (and the variance guide).
/// Everything neural lives in this folder; rt_lab.cpp only calls the functions below.
///

MAKE_COMPUTE_COMPILE_INFO(rt_lab_neural_latent_compile_info, "./src/rendering/rtgi/rt_lab/neural/rt_lab_neural.hlsl", "entry_rt_lab_neural_latent")
// Same pass with the tensor core MLP; only registered when the device has the feature (rt_lab_neural_coopvec_supported).
MAKE_COMPUTE_COMPILE_INFO(rt_lab_neural_latent_coopvec_compile_info, "./src/rendering/rtgi/rt_lab/neural/rt_lab_neural.hlsl", "entry_rt_lab_neural_latent_coopvec")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_neural_atrous_compile_info, "./src/rendering/rtgi/rt_lab/neural/rt_lab_neural.hlsl", "entry_rt_lab_neural_atrous")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_neural_dump_compile_info, "./src/rendering/rtgi/rt_lab/neural/rt_lab_neural.hlsl", "entry_rt_lab_neural_dump")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_neural_ray_stats_compile_info, "./src/rendering/rtgi/rt_lab/neural/rt_lab_neural.hlsl", "entry_rt_lab_neural_ray_stats")

// Pipeline variants of the neural shaders: the feature set (window) and the network precision (half) are compile time defines.
// Variant (false, false) is the plain compile info above; every other combination is registered too (renderer.cpp) and picked at
// runtime from the loaded weights (window) and RtLabSettings::neural_use_half.
inline auto rt_lab_neural_variant(daxa::ComputePipelineCompileInfo2 info, u32 feature_mask, bool half) -> daxa::ComputePipelineCompileInfo2
{
    feature_mask &= RT_LAB_NEURAL_FEATURE_SHADER_BITS;
    if (feature_mask & RT_LAB_NEURAL_FEATURE_WINDOW)
    {
        info.defines.push_back({.name = "RT_LAB_NEURAL_WINDOW", .value = "1"});
    }
    if (feature_mask & RT_LAB_NEURAL_FEATURE_NO_GUIDE_INPUTS)
    {
        info.defines.push_back({.name = "RT_LAB_NEURAL_NO_GUIDE_INPUTS", .value = "1"});
    }
    if (feature_mask & RT_LAB_NEURAL_FEATURE_RAYLENGTH3)
    {
        info.defines.push_back({.name = "RT_LAB_NEURAL_RAYLENGTH3", .value = "1"});
    }
    if (feature_mask & RT_LAB_NEURAL_FEATURE_RAW3)
    {
        info.defines.push_back({.name = "RT_LAB_NEURAL_RAW3", .value = "1"});
    }
    if (feature_mask & RT_LAB_NEURAL_FEATURE_LITE)
    {
        info.defines.push_back({.name = "RT_LAB_NEURAL_LITE", .value = "1"});
    }
    if (feature_mask != 0)
    {
        // Pipeline names are daxa small strings (63 characters, longer names are truncated and then never found in the pipeline map):
        // "<file>::entry_rt_lab_neural_latent_coopvec" is 54 characters, so the suffixes stay short ("::f32" + "::h" = 62).
        info.name += "::f" + std::to_string(feature_mask);
    }
    if (half)
    {
        info.defines.push_back({.name = "RT_LAB_NEURAL_HALF", .value = "1"});
        info.name += "::h";
    }
    return info;
}

// VK_NV_cooperative_vector enabled on the device (daxa ImplicitFeatureFlagBits::COOPERATIVE_VECTOR).
auto rt_lab_neural_coopvec_supported(daxa::Device const & device) -> bool;

// Loads the weights file into the weights buffer when the path changed (or nothing is loaded yet). false = failed
// (state.weights_loaded stays false). Safe to call every frame.
auto rt_lab_neural_ensure_weights(RenderContext & render_context) -> bool;

// Neural passes are part of the graph: the applied preset's neural_enabled and loaded weights.
auto rt_lab_neural_active(RtLabSettings const & settings, RtLabNeuralState const & state) -> bool;

// Start of every frame (rt_lab_frame_update): true = the graph shape (neural on / off, dump on / off) changed since it was
// built. Also loads the weights, allocates the dump buffer and flushes a finished dump to its file.
auto rt_lab_neural_frame_update(RenderContext & render_context) -> bool;

// Every graph build records the neural graph shape (like post_stabilization_built).
void rt_lab_neural_record_built(RtLabSettings const & settings, RtLabNeuralState & state);

// Arms a dump of `frames` frames starting at RUN frame `first_frame` (crop = square at (x, y), clamped to the lab size)
// that is written to `path` once complete; the graph picks it up on its next rebuild.
// frame_list (optional) replaces first_frame / frames: the RUN phase frames to dump.
void rt_lab_neural_dump_begin(RtLabNeuralState & state, std::filesystem::path const & path, i32 first_frame, i32 frames, i32 crop_x, i32 crop_y, i32 crop_size,
    std::vector<i32> frame_list = {});

// Destroys the weights / dump buffers (renderer destruction).
void rt_lab_neural_destroy(RenderContext & render_context);

struct RtLabNeuralLatentViews
{
    daxa::TaskImageView lab_depth = {};
    daxa::TaskImageView lab_face_normal = {};
    daxa::TaskImageView lab_depth_history = {};
    daxa::TaskImageView lab_face_normal_history = {};
    daxa::TaskImageView accumulated = {};
    daxa::TaskImageView lab_raw_diffuse = {};
    daxa::TaskImageView lab_variance = {};
    daxa::TaskImageView lab_ray_length = {};
    // Lite feature set inputs, null = unavailable (zeros): the full res detail normal and the accumulated capped ray length (rt_lab_neural_add_ray_stats_task).
    daxa::TaskImageView detail_normal = {};
    daxa::TaskImageView ray_capped = {};
    daxa::TaskImageView latent_history = {};
    daxa::TaskImageView latent = {};
    daxa::TaskImageView affinity = {};
};
void rt_lab_neural_add_latent_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralLatentViews const & views);

// Per pixel accumulated capped normalized ray length for the lite feature set (needs the lab's ray length buffer = ray length guide on).
struct RtLabNeuralRayStatsViews
{
    daxa::TaskImageView lab_depth = {};
    daxa::TaskImageView lab_face_normal = {};
    daxa::TaskImageView lab_depth_history = {};
    daxa::TaskImageView lab_face_normal_history = {};
    daxa::TaskImageView accumulated = {};
    daxa::TaskImageView lab_raw_diffuse = {};
    daxa::TaskImageView ray_alloc = {};
    daxa::TaskBufferView ray_lengths = {};
    daxa::TaskImageView ray_capped_history = {};
    daxa::TaskImageView ray_capped = {};
};
void rt_lab_neural_add_ray_stats_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralRayStatsViews const & views);

struct RtLabNeuralAtrousViews
{
    daxa::TaskImageView lab_depth = {};
    daxa::TaskImageView lab_face_normal = {};
    daxa::TaskImageView src = {};
    daxa::TaskImageView dst = {};
    daxa::TaskImageView split_src = {};
    daxa::TaskImageView split_dst = {};
    daxa::TaskImageView lab_variance = {};
    daxa::TaskImageView latent = {};
    daxa::TaskImageView affinity = {};
    daxa::TaskImageView history_feedback = {};
};
void rt_lab_neural_add_atrous_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralAtrousViews const & views, u32 pass, bool split_enabled);

struct RtLabNeuralDumpViews
{
    daxa::TaskImageView lab_depth = {};
    daxa::TaskImageView lab_face_normal = {};
    daxa::TaskImageView accumulated = {};
    daxa::TaskImageView lab_raw_diffuse = {};
    daxa::TaskImageView lab_variance = {};
    daxa::TaskImageView lab_ray_length = {};
    // Dump v3 (SH feature set data), null = not available: the temporal SH-Y, CoCg + history length, squared moments of Y, Co, Cg.
    daxa::TaskImageView lab_sh_y = {};
    daxa::TaskImageView lab_cocg = {};
    daxa::TaskImageView lab_sq = {};
    // Dump v2 (raw 3x3 feature set data), null = not available:
    daxa::TaskImageView detail_normal = {};      // full res only
    daxa::TaskImageView half_res_albedo = {};
    daxa::TaskImageView ray_alloc = {};
    daxa::TaskBufferView ray_lengths = {};
};
// Registers the dump buffer and adds the dump task (only call when the dump is enabled).
void rt_lab_neural_add_dump_task(daxa::TaskGraph & tg, RenderContext & render_context, RtLabNeuralDumpViews const & views);
