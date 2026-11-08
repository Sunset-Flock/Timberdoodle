#pragma once

#include "rt_lab.hpp"

///
/// ReTIR: an alternative to ReSTIR GI (mutually exclusive, RtLabSettings::retir_enabled). Temporal reservoir reuse only (no spatial passes), but the
/// reservoir is not collapsed into ONE estimate per pixel: the rays of the pixel are added to the reservoir one at a time (history merge first, the
/// final reservoir is the same as the ReSTIR temporal pass produces) and the estimate after every ray is kept. The denoiser gets all of these
/// intermediate estimates (their mean, the fresh ray count of the pixel as the weight). Uses the trace, shade and filter passes of the ReSTIR stage (rt_lab3.hpp).
///

MAKE_COMPUTE_COMPILE_INFO(rt_lab_retir_temporal_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_retir.hlsl", "entry_rt_lab_retir_temporal")

// Temporal pass of the ReSTIR stage graph (same head as rt_lab3_temporal_callback).
void rt_lab_retir_temporal_callback(daxa::TaskInterface ti, RenderContext * render_context);

// The shared ReSTIR passes read RtLab3Settings: these are ReTIR settings with the spatial reuse off.
auto rt_lab_retir_to_lab3_settings(RtLabRetirSettings const & retir) -> RtLab3Settings;

void rt_lab_retir_ui(RenderContext & render_context);

// Test plan / automation ("retir": {...}): any RtLabRetirSettings field by name. Unknown keys are appended to `unknown_keys`.
void rt_lab_retir_apply_json(RtLabRetirState & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys);
