#pragma once

#include "rt_lab.hpp"

///
/// RT Lab 4: RT Lab 3 bolted in front of RT Lab 2. The ReSTIR GI stage (ray redistribution, trace, temporal + spatial reservoir reuse, ray filter: rt_lab3.hpp)
/// produces one estimate per lab pixel, and those estimates are what the hash grid cells of RT Lab 2 integrate (accumulate, cell integrate, lookup:
/// rt_lab2.hpp). The temporal resolve of RT Lab 2 (its integrator pass, the shared core of rt_lab_integrator.hpp) runs ONCE, at the end. Settings: the ReSTIR
/// stage reads RT Lab 3's, the cells and the temporal resolve RT Lab 2's (no temporal gradient: the cells get no ray records). Half res follows RT Lab 3.
///   ray demand -> distribute -> trace -> reservoirs (temporal, spatial) -> shade -> ray filter  ==>  cells (query, accumulate, integrate) -> lookup -> temporal resolve -> resolve
///

// Start of every frame (rt_lab_frame_update), after RT Lab 3's update (which applies the lab settings overrides for the ReSTIR stage while RT Lab 4 is
// enabled). Decides RtLab4State::active, returns true when the task graph must be rebuilt.
auto rt_lab4_frame_update(RenderContext & render_context, bool reference_phase) -> bool;

auto tasks_rt_lab4_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult;

// UI section of the RT Lab window.
void rt_lab4_ui(RenderContext & render_context);

// Test plan / automation: applies the JSON object ("enabled", "cell_weight_mode") to the state. Unknown keys are appended to `unknown_keys`.
void rt_lab4_apply_json(RtLab4State & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys);
