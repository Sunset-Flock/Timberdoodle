#pragma once

#include "rt_lab.hpp"

///
/// RT Lab 2: a second denoiser for the RT Lab trace (design: rt_lab2.inl). The world space hash grid is the temporal
/// integrator, a stochastic cell interpolation brings the cell values back to pixels, and a small spatial temporal
/// integrator (variance driven history keep) finishes the image. Selected by RtLab2State::enabled; the original
/// denoiser stays untouched and is what the progressive reference phases of the tests run.
///

MAKE_COMPUTE_COMPILE_INFO(rt_lab2_reset_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_reset")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_query_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_query")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_prepare_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_prepare")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_demand_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_demand")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_rays_debug_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_rays_debug")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_stats_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_stats")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_select_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_select")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_distribute_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_distribute")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_gradient_record_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_gradient_record")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_gradient_spatial_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_gradient_spatial")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_gradient_trace_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2_gradient_trace.hlsl", "entry_rt_lab2_gradient_trace")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_accumulate_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_accumulate")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_cell_integrate_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_cell_integrate")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_lookup_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_lookup")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_integrator_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_integrator")
MAKE_COMPUTE_COMPILE_INFO(rt_lab2_probe_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab2.hlsl", "entry_rt_lab2_probe")

// Start of every frame (rt_lab_frame_update), after the preset is applied: decides RtLab2State::active and, when it is
// active, overrides the trace related fields of the preset copy in render_data (full res, no temporal / spatial / guides,
// the ray budget of RT Lab 2). Returns true when the task graph must be rebuilt (the denoiser changed).
auto rt_lab2_frame_update(RenderContext & render_context, bool reference_phase) -> bool;

auto tasks_rt_lab2_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult;

// The cell grid half of RT Lab 2 (exposure, accumulate, cell integrate, lookup, temporal resolve) on a per pixel input (.rgb radiance, .a sample weight),
// used by RT Lab 4. Returns the temporal resolve's output.
auto rt_lab2_add_cell_stage(TasksRtgiInfo const & info, daxa::TaskImageView lab_depth, daxa::TaskImageView lab_face_normal, daxa::TaskImageView lab_raw_diffuse, daxa::Extent3D size) -> daxa::TaskImageView;

// Hash grid settings of a preset (the live copy of its config, see rt_lab_preset_config_ui) / its runtime tools (probes, statistics).
void rt_lab2_ui(RenderContext & render_context);
void rt_lab2_tools_ui(RenderContext & render_context);

// The grid, exposure, lookup and integrator settings of the UI section (RT Lab 4 shows them too).
void rt_lab2_cells_ui(RenderContext & render_context);

// Test plan / automation: applies the JSON object ("enabled", "ray_budget" and any RtLab2Settings field by name) to the
// state. Unknown keys are appended to `unknown_keys`.
void rt_lab2_apply_json(RtLab2State & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys);
