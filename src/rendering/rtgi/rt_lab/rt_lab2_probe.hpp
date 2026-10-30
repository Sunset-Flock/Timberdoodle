#pragma once

#include <array>
#include <deque>
#include <daxa/utils/task_graph.hpp>

#include "../../../timberdoodle.hpp"

using namespace tido::types;

struct RenderContext;

// Live cell probes of RT Lab 2 (UI RT Lab -> RT Lab 2 -> Live Cell Probes, pass entry_rt_lab2_probe in rt_lab2.hlsl). Up to two screen pixels are
// followed every frame and the hash grid cell under each is shown with everything the denoiser knows about it: key, coverage (exposure), rays,
// history samples and its window, the cell's radiance, the pixel's lookup and integrator output, and the 8 trilinear corner taps of the lookup.
// Embedded in RtLab2State::probe (free of RenderContext so rt_lab2_state.hpp can include it). Mirrors the pixel probes (rt_lab_probe.hpp).

#define RT_LAB2_PROBE_UI_COUNT 2
#define RT_LAB2_PROBE_HISTORY_FRAMES 240

struct RtLab2ProbeTap
{
    bool found = false;
    f32 weight = 0.0f;
    f32 samples = 0.0f;
    f32 luminance = 0.0f; // scene units
};

// One frame of one probe (layout of rt_lab2.inl, all radiance in scene units, i.e. RTGI_RADIANCE_SCALE divided out).
struct RtLab2ProbeSample
{
    bool geometry = false;
    bool cell = false;   // the pixel's cell exists
    bool lookup = false; // the lookup found a value for the pixel
    u32 pixel[2] = {};
    u32 cell_index = 0;
    u32 level = 0;
    u32 normal_bucket = 0;
    u32 frames_untouched = 0;
    u32 key = 0;
    i32 coord[3] = {};
    f32 cell_size = 0.0f;
    f32 position[3] = {};
    f32 linear_depth = 0.0f;
    f32 normal[3] = {};
    f32 samples = 0.0f;        // cell history effective sample count
    f32 history[3] = {};       // cell history radiance
    f32 camera_exposure = 1.0f;
    f32 exposure = 0.0f;       // smoothed visible pixel count
    u32 last_pixels = 0;       // pixels that queried the cell last frame
    u32 last_rays = 0;         // rays summed into the cell last frame
    f32 frame_mean[3] = {};    // those rays' mean
    f32 lookup_value[3] = {};
    f32 output[3] = {};        // integrator output
    f32 output_frames = 0.0f;
    // Temporal gradient of the cell.
    f32 gk_x = 0.0f;       // Kalman state: log change per frame
    f32 gk_p = 0.0f;
    f32 gk_applied = 0.0f; // log correction applied to the history this frame
    f32 gk_loss = 0.0f;
    f32 gk_pending = 0.0f;
    f32 meas_z = 0.0f;     // re-trace measurement (valid when meas_r > 0)
    f32 meas_r = 0.0f;
    u32 rec_valid = 0;     // rays recorded last frame (re-traced this frame)
    std::array<RtLab2ProbeTap, 8> taps = {};
};

struct RtLab2ProbeTrend
{
    f32 frame_mean = 0.0f; // luminance, 0 = no rays this frame
    f32 history = 0.0f;
    f32 lookup = 0.0f;
    f32 output = 0.0f;
    f32 samples = 0.0f;
    f32 cap = 0.0f;
    f32 exposure = 0.0f;
    f32 rays = 0.0f;
};

struct RtLab2Probe
{
    bool active = false;    // follows a pixel
    i32 pixel[2] = {};      // screen pixel
    bool pause = false;     // keep showing the last sample
    u64 frames = 0;         // samples received for this pixel
    bool valid = false;     // `latest` holds a sample
    RtLab2ProbeSample latest = {};
    std::deque<RtLab2ProbeTrend> trend = {};
};

struct RtLab2ProbeState
{
    bool enabled = false;   // UI: the pass is part of the graph
    bool built = false;     // graph built with the pass (rebuild trigger)
    daxa::ExternalTaskBuffer buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab2_probe"}}; // host visible ring of slots

    std::array<RtLab2Probe, RT_LAB2_PROBE_UI_COUNT> probes = {};
    i32 picking = -1;       // probe index that takes the next viewport click, -1 = none
    bool exposed_units = false; // luminance x camera exposure x pi (the INDIRECT_DIFFUSE debug view's units)

    // Per frame (rt_lab2_probe_frame_update), read by the dispatch callback.
    bool dispatch_pending = false;
    u32 dispatch_stamp = 0;
    u32 dispatch_active_mask = 0;
    u32 dispatch_pixel[RT_LAB2_PROBE_UI_COUNT][2] = {};

    // Frame stamps: a slot is read once the frames in flight that wrote it are done.
    u32 frame_counter = 0;
    u32 next_stamp = 1;
    u32 last_ingested = 0;
    std::array<u32, 8> issued_stamp = {};
    std::array<u32, 8> issued_frame = {};
};

// UI (every frame while the RT Lab window is open): a viewport click while a probe is picking sets its pixel. Returns true when the click was consumed.
auto rt_lab2_probe_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height) -> bool;
// UI: the two probes side by side.
void rt_lab2_probe_ui(RenderContext & render_context);
// Start of every frame in the renderer: reads finished slots, decides this frame's dispatch. True = the task graph must be rebuilt.
auto rt_lab2_probe_frame_update(RenderContext & render_context) -> bool;
// Renderer destruction.
void rt_lab2_probe_destroy(RenderContext & render_context);
// Task graph dispatch callback (RtLab2ProbeH).
void rt_lab2_probe_callback(daxa::TaskInterface ti, RenderContext * render_context);
