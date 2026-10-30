#pragma once

#include <array>
#include <vector>
#include <daxa/utils/task_graph.hpp>

#include "../../../timberdoodle.hpp"

using namespace tido::types;

struct RenderContext;

// Pixel telemetry (UI RT Lab -> Pixel Telemetry, pass rt_lab_telemetry.hlsl). While the left mouse button is held on a
// pixel, the lab pixel's rays and surface detail of every frame are copied to the CPU and accumulated; the statistics and
// histograms (with a gaussian fit and the two body / tail mixtures of the distribution guide) are computed from the accumulated rays. Embedded in RtLabState::telemetry (free of
// RenderContext so rt_lab_state.hpp can include it).

// One collected ray (layout of rt_lab_telemetry.inl).
struct RtLabTelemetryRay
{
    f32 radiance[3];  // incoming radiance, unmodulated
    f32 hit_t;        // hit distance, >= RT_LAB_RAY_T_MAX / 2 = miss
    f32 modulated[3]; // albedo x radiance x cos(detail normal) / cos(face normal)
    f32 cos_detail;
    f32 direction[3];
    f32 cos_face;
};

struct RtLabTelemetrySurface
{
    bool valid = false; // geometry at the pixel
    u32 screen_pixel[2] = {};
    u32 lab_pixel[2] = {};
    f32 position[3] = {};
    f32 linear_depth = 0.0f;
    f32 face_normal[3] = {};
    f32 detail_normal[3] = {};
    f32 albedo[3] = {};
    f32 metalness = 0.0f;
    f32 pixel_width = 0.0f; // world space width of a lab pixel
    f32 exposure = 1.0f;
    f32 view_direction[3] = {};
};

// Body / tail mixture fits of a luminance series, the batch version of the temporal pass's per pixel tracking (rt_lab_temporal.hlsl,
// rt_lab_temporal_distribution): y = ln(x + eps); the perceptual mean is the mean of y over ALL values, the tail threshold is that + 2 (the
// same offset as RT_LAB_DIST_TAIL_OFFSET, no estimate of the tail feeds back); body = y <= tau, tail = y > tau. Body and tail are lognormal.
// The distribution of x is (1 - p) body + p tail.
struct RtLabTelemetryMixture
{
    bool valid = false;
    f64 eps = 0.0;                // x offset of the log
    f64 perceptual_mean = 0.0;    // mean of y over all values
    f64 tau = 0.0;                // body / tail threshold (y)
    u64 body_count = 0, tail_count = 0;
    f64 tail_fraction = 0.0;      // p
    f64 body_mean = 0.0, body_variance = 0.0;
    f64 tail_mean = 0.0, tail_variance = 0.0;
    // Kolmogorov-Smirnov distance of the whole series (linear values) to the mixture (the gaussian's lives in the fit domain).
    f64 ks_lognormal = 0.0;
};

// Three component version of the mixture fit (the distribution guide mode 3): dark tail + body + bright tail, all lognormal in y = ln(x + eps).
// Same threshold rule as the temporal pass: the dark threshold is the mirror of the bright one (perceptual mean - 2).
struct RtLabTelemetryMixture3
{
    bool valid = false;
    f64 eps = 0.0;
    f64 perceptual_mean = 0.0;
    f64 tau_low = 0.0, tau_high = 0.0;
    u64 low_count = 0, body_count = 0, high_count = 0;
    f64 low_fraction = 0.0, high_fraction = 0.0;
    f64 body_mean = 0.0, body_variance = 0.0;
    f64 low_mean = 0.0, low_variance = 0.0;
    f64 high_mean = 0.0, high_variance = 0.0;
    f64 ks_lognormal = 0.0;
};

// Statistics of one scalar series of the collected rays (luminance of the modulated / unmodulated incoming radiance, hit
// distance).
struct RtLabTelemetrySeries
{
    u64 total = 0;     // values of the series
    u64 count = 0;     // values in the fitted domain
    u64 excluded = 0;  // values left out of the domain (log axis: zeros; hit distance: misses)
    // Linear domain moments (what the pixel actually averages).
    f64 mean = 0.0, variance = 0.0, std_dev = 0.0;
    f64 std_error = 0.0;       // std dev of the mean estimate: sigma / sqrt(n)
    f64 coefficient_of_variation = 0.0; // sigma / mean
    f64 relative_std_error = 0.0;       // std error / mean
    f64 rays_for_5_percent = 0.0;       // rays needed for +-5 % at 95 % confidence: (1.96 CV / 0.05)^2
    f64 min = 0.0, max = 0.0, median = 0.0, p95 = 0.0, p99 = 0.0;
    f64 skewness = 0.0, excess_kurtosis = 0.0;
    f64 effective_sample_ratio = 0.0; // (sum x)^2 / (n sum x^2): 1 = all rays equal, 1/n = one ray carries everything
    f64 top_share = 0.0;              // largest value / sum
    f64 zero_fraction = 0.0;          // values at ~0
    // Fitted gaussian (moments of the fit domain: linear or log10) and its distance to the data.
    f64 fit_mean = 0.0, fit_variance = 0.0, fit_std_dev = 0.0;
    f64 ks_distance = 0.0; // Kolmogorov-Smirnov distance between the fit domain data and the gaussian
    // Body / tail mixtures (luminance series only, valid = enough values).
    RtLabTelemetryMixture mixture = {};
    RtLabTelemetryMixture3 mixture3 = {};
    // Histogram of the fit domain.
    f64 hist_lo = 0.0, hist_hi = 1.0, bin_width = 1.0;
    std::vector<f32> bin_center = {};
    std::vector<f32> bin_count = {};
    u64 below_range = 0, above_range = 0;
    // Mean squared error of the three distributions against the histogram (the plotted bins): per bin the observed ray count vs the count the
    // distribution expects (series values x its probability mass of the bin). [0] gaussian, [1] lognormal body + tail mixture, [2] dark tail + lognormal body + bright tail.
    // mse_abs = mean over the bins of (observed - expected)^2 (rays^2), mse_rel = sum (observed - expected)^2 / sum observed^2 (0 = perfect).
    bool mse_valid[3] = {};
    f64 mse_abs[3] = {};
    f64 mse_rel[3] = {};
    bool reference = false; // the data of this series (bars, statistics, KS, MSE) are the reference rays; the distributions are fitted on the collected rays
    bool log_domain = false; // the histogram + gaussian fit live in log10 of the values (the Log Axis option, or Auto Histogram chose it)
    bool valid = false;
};

struct RtLabTelemetryStats
{
    RtLabTelemetrySeries modulated = {};   // luminance of the cos + albedo + normal modulated radiance
    RtLabTelemetrySeries incoming = {};    // luminance of the unmodulated incoming radiance
    RtLabTelemetrySeries ray_length = {};  // hit distance (hits only)
    u64 rays = 0;
    u64 frames = 0;
    f64 mean_rays_per_frame = 0.0;
    f64 miss_fraction = 0.0;      // rays that escaped (sky visibility)
    f64 near_hit_fraction = 0.0;  // rays that hit within 1 m
    f64 mean_modulated[3] = {};   // mean rgb of the modulated radiance (outgoing diffuse radiance)
    f64 mean_incoming[3] = {};    // mean rgb of the unmodulated incoming radiance
    f64 mean_cos_detail = 0.0;
    f64 mean_cos_face = 0.0;
    f64 mean_importance_weight = 0.0; // mean of cos(detail) / cos(face)
    f64 direction_concentration = 0.0; // |luminance weighted mean direction| / 1: 0 = light from everywhere, 1 = one direction
    f64 mean_direction[3] = {};        // normalized luminance weighted mean direction
    f64 mean_direction_angle = 0.0;    // degrees between it and the detail normal
    f64 detail_face_angle = 0.0;       // degrees between detail and face normal
    f64 view_cos = 0.0;                // dot(-view direction, detail normal)
};

struct RtLabTelemetryState
{
    bool enabled = false;          // UI: the pass is part of the graph and the mouse collects rays
    bool built = false;            // graph built with the pass (rebuild trigger)
    daxa::ExternalTaskBuffer buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab_telemetry"}}; // host visible ring of slots

    // Options (UI).
    bool log_axis = false;       // histograms + gaussian fit over log10 of the values
    bool exposed_units = false;  // luminance in exposed irradiance units (x exposure x pi, the INDIRECT_DIFFUSE view's units)
    bool use_collected = false;  // histogram + statistics of the collected rays instead of the reference rays (the fits are always the collected rays')
    bool auto_histogram = true;  // bin count (Freedman-Diaconis on the data's spread) adjusts to the collected data
    i32 bins = 48;               // used with auto_histogram off
    i32 max_samples = 4000;      // rays kept (collection stops when full)

    // Input of this frame (rt_lab_telemetry_input), consumed by rt_lab_telemetry_frame_update.
    bool collecting = false;
    i32 pixel[2] = {};
    bool was_collecting = false;
    i32 target_pixel[2] = {};

    // Per frame (rt_lab_telemetry_frame_update), read by the dispatch callback.
    bool dispatch_pending = false;
    u32 dispatch_stamp = 0;
    u32 dispatch_pixel[2] = {};

    // Frame stamps: a slot is read once the frames in flight that wrote it are done.
    u32 frame_counter = 0;
    u32 next_stamp = 1;
    u32 first_stamp = 1;   // stamps below belong to an earlier press / pixel
    u32 last_ingested = 0;
    std::array<u32, 8> issued_stamp = {};
    std::array<u32, 8> issued_frame = {};

    // Reference rays: RT_LAB_TELEMETRY_REFERENCE_RAYS rays of the pixel, traced in ONE frame when the button is released (never part of the
    // denoiser). The histogram, statistics, KS and MSE use them, the distributions are fitted on the collected rays.
    daxa::ExternalTaskBuffer reference_buffer = daxa::ExternalTaskBuffer{{.name = "rt_lab_telemetry_reference"}};
    bool reference_dispatch_pending = false; // this frame's graph traces the reference rays
    bool reference_in_flight = false;
    u32 reference_stamp = 0;
    u32 reference_issued_frame = 0;
    u32 reference_screen_pixel[2] = {};
    bool reference_valid = false;
    std::vector<RtLabTelemetryRay> reference_rays = {};

    // Collected data.
    std::vector<RtLabTelemetryRay> rays = {};
    u64 frames = 0;
    u64 rays_seen = 0;       // rays of the pixel the frames had, collected or not
    bool full = false;
    RtLabTelemetrySurface surface = {};
    bool stats_dirty = true;
    u32 stats_age = 0;
    RtLabTelemetryStats stats = {};
};

// Density of x (>= 0) under the three component mixture (shared with the live pixel probes).
auto rt_lab_telemetry_mixture3_pdf(RtLabTelemetryMixture3 const & m, f64 x) -> f64;

// UI (every frame while the RT Lab window is open): the mouse input. mouse_down = left button held on the viewport.
void rt_lab_telemetry_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height);
// UI: settings, surface detail, statistics, histograms.
void rt_lab_telemetry_ui(RenderContext & render_context);
// Start of every frame in the renderer: reads finished slots, decides this frame's dispatch. True = the task graph must be rebuilt.
auto rt_lab_telemetry_frame_update(RenderContext & render_context) -> bool;
// Renderer destruction.
void rt_lab_telemetry_destroy(RenderContext & render_context);
// Task graph dispatch callback (RtLabTelemetryH).
void rt_lab_telemetry_callback(daxa::TaskInterface ti, RenderContext * render_context);
// Task graph dispatch callback (RtLabTelemetryReferenceH): the reference rays of the released pixel.
void rt_lab_telemetry_reference_callback(daxa::TaskInterface ti, RenderContext * render_context);
