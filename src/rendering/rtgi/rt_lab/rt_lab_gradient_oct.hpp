#pragma once

#include "rt_lab.hpp"

///
/// Temporal gradient 2: the samples of the temporal gradient re-trace as a pyramid of small octahedral maps (old + new
/// radiance per direction, with second moments), position gated, optionally denoised and cosine convolved, resolved to a per
/// pixel result (the shrunk change of the lighting at the pixel's normal) that can scale the temporal history.
/// See rt_lab_gradient_oct.inl.
///

MAKE_COMPUTE_COMPILE_INFO(rt_lab_gradient_oct_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_gradient_oct.hlsl", "entry_rt_lab_gradient_oct")
MAKE_COMPUTE_COMPILE_INFO(rt_lab_gradient_oct_resolve_compile_info, "./src/rendering/rtgi/rt_lab/rt_lab_gradient_oct.hlsl", "entry_rt_lab_gradient_oct_resolve")

// Adds the pyramid passes (positions, bin, denoise, convolve per level, then the per pixel resolve + the debug draw). The passes only
// run while the history scale (temporal_gradient2_scale_history) is on or one of the DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_* /
// _GRADIENT2_* views is active. oct_samples / oct_positions = the re-trace's per cell samples and the world positions of their
// surface points (cell_size), the g-buffer geometry comes from info (the resolve runs before the ray demand).
// Returns the per pixel result (lab resolution, RGBA16F): .x = the shrunk log change of the lighting (ln new / old) at the pixel's
// normal, .y = the remaining bias variance, .z = the level that won, .w = 1 valid / 0 no information.
auto rt_lab_gradient_oct_add_tasks(TasksRtgiInfo const & info, daxa::TaskImageView oct_samples, daxa::TaskImageView oct_positions, daxa::Extent3D cell_size, daxa::Extent3D lab_size) -> daxa::TaskImageView;
