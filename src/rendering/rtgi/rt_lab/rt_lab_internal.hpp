#pragma once

#include <cstring>
#include <string>

#include "rt_lab.hpp"

// Task callbacks of the original RT Lab (rt_lab.cpp) that RT Lab 2 (rt_lab2.cpp) reuses as they are: the trace
// (ray demand -> distribute -> trace -> gather), the resolve, the measurement and the reference comparison.

// Pointer style push (attach = device address of the attachment blob), push.size = size_image's size, one thread per
// size_image pixel.
template <typename Push>
inline void rt_lab_dispatch(daxa::TaskInterface ti, RenderContext * render_context, Push push, daxa::TaskImageAttachmentIndex size_image, u32 block_size, u32 render_time, std::string const & pipeline_name)
{
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, render_time);
    auto const size = ti.info(size_image).value().size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(pipeline_name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    push.size = {size.x, size.y};
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(size.x, block_size), round_up_div(size.y, block_size), 1});
}

void rt_lab_demand_callback(daxa::TaskInterface ti, RenderContext * render_context, bool history_bound, bool gradient_bound);
void rt_lab_distribute_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity);
void rt_lab_trace_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity, bool ray_lengths_bound);
void rt_lab_ray_filter_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity);
void rt_lab_gather_callback(daxa::TaskInterface ti, RenderContext * render_context, bool gradient_enabled, bool filtered_bound, bool gradient_per_pixel);
void rt_lab_resolve_callback(daxa::TaskInterface ti, RenderContext * render_context, bool split_enabled, bool spatial_ran);
void rt_lab_convergence_capture_callback(daxa::TaskInterface ti, RenderContext * render_context, bool split_enabled);
void rt_lab_convergence_analyze_callback(daxa::TaskInterface ti, RenderContext * render_context);
void rt_lab_measure_accumulate_callback(daxa::TaskInterface ti, RenderContext * render_context);
void rt_lab_measure_lag_callback(daxa::TaskInterface ti, RenderContext * render_context);
void rt_lab_measure_store_callback(daxa::TaskInterface ti, RenderContext * render_context);

// Measurement + reference comparison passes of the tests on the resolved image (rt_lab.cpp), for the labs without a split history (RT Lab 2 / 3 / 4).
void rt_lab_add_test_passes(TasksRtgiInfo const & info, daxa::TaskImageView resolved, daxa::Extent3D size);
