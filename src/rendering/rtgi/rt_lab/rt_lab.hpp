#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include <nlohmann/json_fwd.hpp>

#include "../../scene_renderer_context.hpp"
#include "../rtgi.hpp" // TasksRtgiInfo / TasksRtgiMainResult: RT Lab is a drop-in replacement for tasks_rtgi_main

///
/// RT Lab: experimental diffuse RTGI path, selected instead of the RTGI denoiser by rt_lab_settings.enabled.
/// Reuses the RTGI ray shading core (rtgi_shade.hlsl) and shared helpers; the denoiser passes are its own.
///
/// Current pipeline (full res by default, rt_lab_settings.half_res for half res):
///     ray demand (lab geometry + desired rays from the reprojected history length) -> distribute (hard budget, ray list)
///     -> trace (1 cosine diffuse ray per ray list entry, inline ray query + RTGI coarse shading) -> gather (per pixel mean)
///     (temporal gradient: re-trace of last frame's 1-in-16 records + blur before the demand, lambda shrinks the history)
///     -> temporal accumulation (reprojection + geometric disocclusion, "smart" fp16 or "progressive" fp32 history)
///     -> separable bilateral blur (horizontal, vertical; geometric edge stopping only)
///     -> resolve (full res output, bilateral upsample at half res)
///     -> convergence error estimate capture + analysis (only once an estimate was started, rt_lab_convergence.hlsl)
/// Settings: two presets (progressive reference, custom) in RenderContext::rt_lab (rt_lab_state.hpp).
/// No specular, no firefly filter, no guides.
///

auto rt_lab_default_settings(RtLabPreset preset) -> RtLabSettings;
// base with the preset fields in `overrides` replaced (keys = the saved result settings keys, e.g. "coarse_anti_lag_enabled").
// Unknown keys are appended to `unknown_keys` and ignored. Non preset fields (enabled, reset_history, ...) stay base's.
auto rt_lab_settings_apply_json(RtLabSettings const & base, nlohmann::json const & overrides, std::vector<std::string> & unknown_keys) -> RtLabSettings;

// Fills the presets with their defaults (renderer construction).
void rt_lab_init(RenderContext & render_context);
// Destroys the convergence estimate capture images (renderer destruction).
void rt_lab_destroy(RenderContext & render_context);
// Every frame in the application update, before the camera is updated: runs the test state machines (convergence error
// estimate, convergence measurement) and decides this frame's programmed move pose (RtLabState::test_move / _step).
void rt_lab_tests_begin_frame(RenderContext & render_context);
// Start of every frame in the renderer, before the graph rebuild check: copies the active (or test) preset into
// render_data.rt_lab_settings and collects the estimate readback. Returns true when the task graph must be rebuilt.
auto rt_lab_frame_update(RenderContext & render_context) -> bool;
// Steps of a test's programmed move: the longer of the camera and the sun move, in fixed_dt steps.
auto rt_lab_test_move_steps(RtLabTestMove const & move) -> i32;
// The application took the step 0 poses of a starting test move (RtLabState::test_move_begin): recorded with the test.
void rt_lab_set_test_move_origins(RtLabState & lab, ProgrammedMovePose const & camera, ProgrammedMovePose const & sun);

MAKE_COMPUTE_COMPILE_INFO(rt_lab_demand_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_rays.hlsl", "entry_rt_lab_demand")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_distribute_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_rays.hlsl", "entry_rt_lab_distribute")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_ray_filter_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_rays.hlsl", "entry_rt_lab_ray_filter")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_gather_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_rays.hlsl", "entry_rt_lab_gather")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_gradient_blur_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_rays.hlsl", "entry_rt_lab_gradient_blur")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_gradient_trace_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_trace.hlsl", "entry_rt_lab_gradient_trace")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_trace_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_trace.hlsl", "entry_rt_lab_trace")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_anti_lag_downsample_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_anti_lag.hlsl", "entry_rt_lab_anti_lag_downsample")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_anti_lag_blur_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_anti_lag.hlsl", "entry_rt_lab_anti_lag_blur")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_temporal_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_temporal.hlsl", "entry_rt_lab_temporal")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_blur_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_blur.hlsl", "entry_rt_lab_blur")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_coarse_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_coarse.hlsl", "entry_rt_lab_coarse")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_atrous_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_blur.hlsl", "entry_rt_lab_atrous")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_resolve_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_resolve.hlsl", "entry_rt_lab_resolve")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_convergence_capture_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_convergence.hlsl", "entry_rt_lab_convergence_capture")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_convergence_analyze_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_convergence.hlsl", "entry_rt_lab_convergence_analyze")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_measure_accumulate_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_measure.hlsl", "entry_rt_lab_measure_accumulate")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_measure_lag_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_measure.hlsl", "entry_rt_lab_measure_lag")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_measure_store_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_measure.hlsl", "entry_rt_lab_measure_store")

// == Convergence measurement analytics (RtLabMeasurement, rt_lab_state.hpp). Metric values are fractions (0.01 = 1 %).
// Lower edge of histogram bin k of m (k = RT_LAB_MEASURE_BINS: upper edge of the last bin).
auto rt_lab_measure_bin_edge(RtLabMeasurement const & m, u32 k) -> f64;
// Fraction (0..1) of the tiles whose metric is below `value` at `frame` (resolution: one bin).
auto rt_lab_measure_fraction_below(RtLabMeasurement const & m, i32 metric, i32 frame, f64 value) -> f64;
// Metric value at the `fraction` percentile at `frame`. Infinity: that many tiles never got below the histogram range.
auto rt_lab_measure_percentile(RtLabMeasurement const & m, i32 metric, i32 frame, f64 fraction) -> f64;
// Temporal lag (step response) of a measurement with lag data at a relative tolerance: for the tiles that changed by
// more than 2 x tolerance after m.lag_start_frame, the frames after it until each stays within the tolerance of its
// final value. Returns m.lag for measurements without lag data (loaded from files).
auto rt_lab_measure_lag(RtLabMeasurement const & m, f64 tolerance) -> RtLabLagSummary;
// Frames of accumulation after which `fraction` of the tiles are below `value` and STAY there until the end of the run,
// -1 = not reached.
auto rt_lab_measure_frames_to_reach(RtLabMeasurement const & m, i32 metric, f64 value, f64 fraction) -> i32;

// == Measurement results (RtLabState::results) ==
// Adds every saved result in RT_LAB_RESULTS_DIRECTORY not in the list yet (metadata only).
void rt_lab_results_scan(RtLabState & lab);
// Saves results[index] as RT_LAB_RESULTS_DIRECTORY/<name>.rtlab (overwrites), renames the entry. false = failed.
auto rt_lab_result_save(RtLabState & lab, i32 index, std::string const & name) -> bool;
// Renames results[index]. Saved results are re-saved as <name>.rtlab and the old file is removed. false = failed.
auto rt_lab_result_rename(RtLabState & lab, i32 index, std::string const & name) -> bool;
// Removes results[index] from the list; a saved result's file is deleted too. false = the file could not be deleted.
auto rt_lab_result_delete(RtLabState & lab, i32 index) -> bool;
// Loads a saved result's histogram rows if not loaded yet. false = failed.
auto rt_lab_result_ensure_loaded(RtLabResult & result) -> bool;

// Graph shape settings (task graph rebuild trigger).
// Ray list entries per lab pixel (rt_lab_rays.inl): the budget rounded up, 1 without redistribution.
inline auto rt_lab_ray_capacity_per_pixel(RtLabSettings const & s) -> u32
{
    return s.redistribution_enabled != 0 ? static_cast<u32>(std::max(std::ceil(s.ray_budget), 1.0f)) : 1u;
}

inline auto rt_lab_settings_changed(RtLabSettings const & a, RtLabSettings const & b) -> bool
{
    return a.enabled != b.enabled ||
           rt_lab_ray_capacity_per_pixel(a) != rt_lab_ray_capacity_per_pixel(b) ||
           a.temporal_gradient_enabled != b.temporal_gradient_enabled ||
           a.half_res != b.half_res ||
           a.temporal_enabled != b.temporal_enabled ||
           a.temporal_progressive != b.temporal_progressive ||
           a.split_history_enabled != b.split_history_enabled ||
           a.spatial_enabled != b.spatial_enabled ||
           a.spatial_mode != b.spatial_mode ||
           a.atrous_passes != b.atrous_passes ||
           a.coarse_enabled != b.coarse_enabled ||
           a.coarse_atrous_passes != b.coarse_atrous_passes ||
           ((a.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MODE_MASK) != RT_LAB_ANTI_LAG_OFF) != ((b.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MODE_MASK) != RT_LAB_ANTI_LAG_OFF) ||
           (a.ray_filter_mode != RT_LAB_RAY_FILTER_OFF) != (b.ray_filter_mode != RT_LAB_RAY_FILTER_OFF);
}

// Opaque specular is null: the opaque shading then falls back to its diffuse based specular approximation.
auto tasks_rt_lab_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult;
