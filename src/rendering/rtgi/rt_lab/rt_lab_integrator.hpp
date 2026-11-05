#pragma once

#include "rt_lab.hpp"
#include "rt_lab_integrator_state.hpp"

///
/// Temporal resolve shared by all RT labs (design: rt_lab_integrator.inl). Run it on a lab's final per pixel radiance, right before the resolve:
///   * tasks: `lab_diffuse = rt_lab_add_integrator(...)` when state.enabled, the resolve reads the returned image
///   * UI: rt_lab_integrator_ui(state) inside the lab's own section
///   * automation: rt_lab_integrator_apply_json(state, json, unknown)
/// Each lab owns its RtLabIntegratorState (RtLabState::integrator for the original lab, RtLab3State::integrator for RT Lab 3). RT Lab 2 runs the same
/// shader core (rt_lab_integrate) inside its own integrator pass, with its temporal gradient correction.
///

MAKE_COMPUTE_COMPILE_INFO(rt_lab_integrator_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_integrator.hlsl", "entry_rt_lab_integrator")

// Adds the pass: input (.rgb RTGI_RADIANCE_SCALE units, .a > 0 = valid) -> persistent output (.rgb radiance, .a history length frames), returned as
// the current image. lab_depth / lab_face_normal are the lab's PERSISTENT_DOUBLE_BUFFER geometry images (the pass reprojects against .previous()).
auto rt_lab_add_integrator(
    TasksRtgiInfo const & info,
    RtLabIntegratorState & state,
    daxa::TaskImageView lab_depth,
    daxa::TaskImageView lab_face_normal,
    daxa::TaskImageView input,
    daxa::Extent3D size,
    char const * history_name) -> daxa::TaskImageView;

// Start of the frame: true when the task graph must be rebuilt (the pass was switched on / off).
auto rt_lab_integrator_frame_update(RtLabIntegratorState & state) -> bool;

// Checkbox + settings, to be called inside the lab's own UI section (unique id per instance).
void rt_lab_integrator_ui(RtLabIntegratorState & state, RenderContext & render_context, char const * id);

// Applies the JSON object ("enabled" + any RtLabIntegratorSettings field by name). Unknown keys are appended to `unknown_keys`.
void rt_lab_integrator_apply_json(RtLabIntegratorState & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys);
