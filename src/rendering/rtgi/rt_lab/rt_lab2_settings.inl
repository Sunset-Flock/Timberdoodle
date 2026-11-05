#pragma once

#include "daxa/daxa.inl"

#define RT_LAB2_CELL_CAPACITY_LOG2_DEFAULT 20 // RtLab2Settings::cell_capacity_log2 (0 = this)

// RT Lab 2 settings (rt_lab2.inl documents the pipeline). Plain shared types: RtLab2State (rt_lab2.hpp) holds them on the CPU,
// every pass reads them through a pointer in its push constant.

// Lookup kernels: every tap draws a random offset (per axis, in cells of the tap's cascade, scaled by lookup_distance) from the kernel
// distribution and reads the cell that contains the point. No weights, no deterministic taps: the mean of the taps is the filtered value.
#define RT_LAB2_KERNEL_TRILINEAR 0 // tent: triangular distribution over +-1 cell
#define RT_LAB2_KERNEL_CUBIC 1     // cubic B-spline: sum of 4 uniforms, +-2 cells (sigma 0.58), smoother than the tent
#define RT_LAB2_KERNEL_GAUSSIAN 2  // gaussian, sigma 0.75 cells
#define RT_LAB2_MAX_LOOKUP_TAPS 32
#define RT_LAB2_MAX_WINDOW_RADIUS 4

// Settings (CPU: RtLab2State::settings), handed to every pass through a pointer in the push constant.
struct RtLab2Settings
{
    // == Grid ==
    daxa_f32 cell_size_px;          // target cell size in pixels at the cell's distance (picks the cascade level)
    daxa_f32 min_cell_size;         // world meters, level 0
    // Stochastic cascade blend: around every level transition (in log2 cell size units, 0 = hard levels, 1 = the whole level range) a tap picks the
    // neighbor cascade with a probability that is 0.5 exactly at the transition and falls to 0 at the edge of the region. The pixels in the
    // region feed both cascades' cells (query + accumulate), so both hold data.
    daxa_f32 level_blend_width;
    daxa_i32 max_untouched_frames;  // a cell no pixel queried for this long is freed
    // Temporal adaption speed, normalized by the pixel coverage: a cell seen by fewer pixels gets proportionally fewer rays, so its history window
    // (the cap of the effective sample count) shrinks the same way: cap = history_samples_per_pixel x the cell's smoothed visible pixel count,
    // clamped to [min_history_samples, max_history_samples]. The blend of a frame n_new / (n_old + n_new) then only depends on rays per covered
    // pixel, not on how large the cell looks on screen. history_samples_per_pixel 0 = off (the cap is max_history_samples).
    daxa_f32 history_samples_per_pixel;
    daxa_f32 min_history_samples;
    daxa_f32 max_history_samples;   // upper limit of the cap
    daxa_f32 max_frame_samples;     // cap of the weight one frame's rays / pixels get in a cell's blend n_new / (n_old + n_new), 0 = off (RT Lab 4: a frame cannot pull a cell harder than this)
    daxa_f32 cell_capacity_log2;    // log2 of the number of cell slots (the grid's memory). 0 = the default 20 (this slot was the removed pixel clamp: older builds send 0)
    // Exposure = how many pixels see the cell (pixel coverage). exposure itself is smoothed (the history window follows it); the weight of a new frame
    // against the history follows the raw coverage change between the last frame and this one (exposure_strength).
    daxa_f32 exposure_alpha;        // smoothing rate of the visible pixel count
    daxa_f32 exposure_strength;     // 0 = off; history samples x (last frame's coverage / this frame's coverage)^strength, both directions (1 = 4 -> 16 pixels: the new frame counts 4x more)
    daxa_f32 exposure_min_scale;    // limits of one frame's scaling: [min scale, 1 / min scale]
    // == Ray redistribution ==
    // The frame's rays (RtLab2State::ray_budget, hard) are spread by the cells: a pixel asks for 1 base ray plus extras that fall off exponentially with how
    // converged its cell is (RTGI demand curve, exp(-4 x), x = cell samples / (redistribution_target x the cell's history window)), up to
    // max_rays_per_pixel. Fresh / disoccluded / changed cells (their samples were scaled down) get the rays, converged cells give them up. The
    // distribute pass drains the extras first, down to the base coverage (RtLab2State::min_base_coverage). Off: 1 ray per selected pixel.
    daxa_i32 redistribution_enabled;
    daxa_i32 max_rays_per_pixel;      // base included
    daxa_f32 redistribution_target;   // fraction of the history window at which a cell stops asking for extra rays
    // == Lookup ==
    daxa_i32 lookup_kernel;         // RT_LAB2_KERNEL_*
    daxa_i32 lookup_taps;           // stochastic taps per pixel and frame (the integrator averages the noise)
    daxa_f32 lookup_distance;       // kernel width scale: 1 = the kernel's own width in cells, > 1 wider (smoother), < 1 narrower
    daxa_i32 lookup_project;        // 1 = tap offsets are projected onto the pixel's surface plane
    daxa_i32 lookup_min_samples;    // taps on cells with fewer effective samples are skipped
    // == Integrator ==
    daxa_i32 integrator_enabled;    // 0 = the lookup is the output
    daxa_i32 window_radius;         // variance window (2r + 1)^2 on the lookup of this frame
    daxa_f32 variance_gamma;        // neighborhood box half width (window std devs) the history is clamped into, 0 = no clamp
    daxa_f32 alpha_min;             // blend factor of the new value at the noisiest windows (keeps most history)
    daxa_f32 alpha_max;             // ... at quiet windows (trusts the new value)
    daxa_f32 variance_reference;    // window relative std at which alpha reaches alpha_min
    daxa_f32 spatial_strength;      // 0..1: how far a noisy window moves the new value to the window mean
    daxa_i32 max_frames;            // cap of the integrator history length (alpha never goes below 1 / length)
    daxa_f32 plane_tolerance;       // reprojection / window surface test in pixel widths
    daxa_f32 normal_threshold;      // reprojection / window face normal dot threshold
    // == Temporal gradient (per cell) ==
    // Every cell remembers up to RT_LAB2_GRADIENT_RECORDS of its fresh rays each frame (all of them at gradient_full_coverage pixels of coverage, fewer below) and
    // re-traces them the next frame bit identically (same quantized direction + hit seed): the change of their luminance is a measurement z = ln(new / old) of how
    // the lighting changed (R from their spread, at least gradient_min_variance). A spatial pass over one candidate pixel of every 2x2 quad blurs the measurements of the
    // cells around it (stochastic taps, geometry aware, precision weighted) and integrates the result of all the pixels of a cell into ONE gradient value of the cell.
    // The cell runs a Kalman filter on it (innovation gate = reset on shocks), takes the Wiener shrunk correction c (with glide) and applies it: history x exp(c), and
    // its sample count N' = N / (1 + N k L / v) with L the uncertainty that is left.
    daxa_i32 gradient_enabled;
    daxa_f32 gradient_full_coverage;       // pixels of coverage at which a cell records the maximum number of rays
    daxa_i32 gradient_blur_taps;           // stochastic taps of the spatial blur per half res pixel (+ the center)
    daxa_f32 gradient_blur_radius;         // px, the taps' disc
    daxa_f32 gradient_min_variance;        // floor of a measurement's variance (log ratio^2)
    daxa_f32 gradient_kalman_q;            // process noise per frame
    daxa_f32 gradient_kalman_gate;         // innovation gate in standard deviations: beyond it the filter resets to the measurement
    daxa_f32 gradient_cap_k;               // k of the sample count cut
    daxa_f32 gradient_sample_variance;     // gain of the S^4 term of the sample count cut (large changes cut hard, small drifts lightly)
    daxa_f32 gradient_glide;               // fraction of a moderate correction applied per frame (the rest stays pending), 1 = off
    daxa_i32 gradient_correct_integrator;  // the integrator's reprojected history is scaled by the cell's applied correction too
    // == Frame ==
    daxa_i32 reset;                 // reset the history this frame
    daxa_i32 animate_noise;
};
DAXA_DECL_BUFFER_PTR(RtLab2Settings)

