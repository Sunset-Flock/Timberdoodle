#pragma once

#include "rt_lab.hpp"

///
/// RT Lab 3: ReSTIR GI on the RT Lab trace (design: rt_lab3.inl). Ray redistribution (the old lab's demand + distribute), a trace that stores
/// hit position + normal + radiance, temporal + spatial reservoir reuse, a ray filter on the final estimate, the old lab's resolve (also half
/// res). Selected by RtLab3State::enabled; the original denoiser stays untouched and is what the progressive reference phases of the tests run.
///

MAKE_COMPUTE_COMPILE_INFO(rt_lab3_trace_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab3.hlsl", "entry_rt_lab3_trace")
MAKE_COMPUTE_COMPILE_INFO(rt_lab3_temporal_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab3.hlsl", "entry_rt_lab3_temporal")
MAKE_COMPUTE_COMPILE_INFO(rt_lab3_spatial_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab3.hlsl", "entry_rt_lab3_spatial")
MAKE_COMPUTE_COMPILE_INFO(rt_lab3_shade_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab3.hlsl", "entry_rt_lab3_shade")
MAKE_COMPUTE_COMPILE_INFO(rt_lab3_filter_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab3.hlsl", "entry_rt_lab3_filter")

// Start of every frame (rt_lab_frame_update), after the preset is applied: decides RtLab3State::active and, when it is active, overrides
// the trace related fields of the preset copy in render_data (lab resolution, ray budget, no temporal / spatial / guides of the old lab).
// Returns true when the task graph must be rebuilt (the denoiser or the spatial pass count changed).
auto rt_lab3_frame_update(RenderContext & render_context, bool reference_phase) -> bool;

auto tasks_rt_lab3_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult;

// The ReSTIR stage of the graph (ray demand, distribute, trace, temporal + spatial reuse, shade, ray filter), without the temporal resolve / resolve / tests:
// RT Lab 3 finishes it, RT Lab 4 feeds it into the cell grid. cell_weight_mode >= 0 (RT_LAB3_CELL_WEIGHT_*) also writes `cell_input` (the filtered estimate
// with that per pixel weight in .a). cell_source: RT_LAB3_CELL_SOURCE_*.
struct RtLab3Stage
{
    daxa::TaskImageView lab_depth = {};       // PERSISTENT_DOUBLE_BUFFER geometry of the lab resolution
    daxa::TaskImageView lab_face_normal = {};
    daxa::TaskImageView output = {};          // current: filtered estimate, .a = temporal reservoir confidence M
    daxa::TaskImageView cell_input = {};      // only with cell_weight_mode >= 0
    daxa::TaskImageView ray_alloc = {};       // per pixel ray list offset / count of this frame
    daxa::Extent3D size = {};                 // lab resolution
    daxa::Extent3D full_size = {};
};
auto rt_lab3_add_restir(TasksRtgiInfo const & info, i32 cell_weight_mode, i32 cell_source) -> RtLab3Stage;

// UI section of the RT Lab window.
void rt_lab3_ui(RenderContext & render_context);

// The settings of the section without the enable checkbox (RT Lab 4 shows them for its ReSTIR stage).
void rt_lab3_settings_ui(RenderContext & render_context);

// Test plan / automation: applies the JSON object ("enabled", "half_res", "ray_budget", ... and any RtLab3Settings field by name) to the
// state. Unknown keys are appended to `unknown_keys`.
void rt_lab3_apply_json(RtLab3State & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys);
