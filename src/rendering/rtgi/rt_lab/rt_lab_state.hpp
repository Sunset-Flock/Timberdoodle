#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <daxa/utils/task_graph.hpp>

#include "../../../timberdoodle.hpp"
#include "../../../programmed_move.hpp"
#include "../../../shader_shared/readback.inl"
#include "rt_lab_settings.inl"
#include "rt_lab_measure_layout.inl"

using namespace tido::types;

// CPU side RT Lab state, owned by RenderContext (render_context.rt_lab). rt_lab_frame_update (rt_lab.hpp) applies it
// to RenderGlobalData::rt_lab_settings at the start of every frame.

enum RtLabPreset : i32
{
    RT_LAB_PRESET_PROGRESSIVE = 0, // reference: full res, fp32 long accumulation, 2 tight a-trous passes, no coarse fallback
    RT_LAB_PRESET_CUSTOM = 1,      // the settings under test
    RT_LAB_PRESET_COUNT = 2,
};

enum struct RtLabConvergencePhase
{
    IDLE,
    PROGRESSIVE, // accumulating the progressive preset reference
    CUSTOM,      // accumulating the custom preset
    // With a programmed move (RtLabTestMove): every step k of the move, custom reprojects its history to pose k and adds
    // ONE frame (captured), then progressive resets and accumulates move_reference_frames frames at pose k (captured +
    // compared on its last frame). Custom's history survives the progressive phases through backup copies.
    MOVE_WARMUP,      // custom accumulating at step 0 (from a reset) before the move, captured on its last frame
    MOVE_CUSTOM,      // custom: one frame at the current step
    MOVE_PROGRESSIVE, // progressive reference at the current step
    MOVE_COLLECTING,  // waiting for the last step's analysis readback
};

// Programmed move of a test (sub option of both tests): camera and sun move, stepped with a FIXED time step (also the
// frame delta time while the test runs) so runs are reproducible. Step k pose = programmed_move_pose_at(origin, k).
struct RtLabTestMove
{
    bool enabled = false;
    f32 fixed_dt = 1.0f / 60.0f;
    ProgrammedMove camera = {.duration = 2.0f};
    ProgrammedMove sun = {.translation = false, .duration = 2.0f};
};

// Error of one programmed move step of the convergence error estimate (custom after one reprojected frame vs the fresh
// progressive reference at the same pose).
struct RtLabConvergenceStep
{
    i32 step = 0;
    u32 pixels = 0;
    f64 mse = 0.0;
    f64 bias = 0.0;
    f64 mean_abs_error = 0.0;
    f64 rms_predicted = 0.0; // split history noise prediction, 0 = none
    f64 rms_actual = 0.0;
};

enum struct RtLabMeasurePhase
{
    IDLE,
    RUNNING,    // recording one histogram row per frame
    COLLECTING, // waiting for the last rows' frames to finish on the GPU
};

// Temporal lag summary of a measurement (rt_lab_measure_lag): tiles that changed by more than 2 x tolerance after
// start_frame, frames until each stays within tolerance of its final value (relative, final = mean of the last
// RT_LAB_MEASURE_LAG_FINAL_FRAMES frames).
struct RtLabLagSummary
{
    bool valid = false;
    f64 tolerance = 0.02;
    i32 start_frame = 0;
    u32 tiles = 0;          // tiles with geometry
    u32 changed_tiles = 0;  // tiles that changed by more than 2 x tolerance
    f64 settle_p50 = 0.0;   // frames after start_frame
    f64 settle_p90 = 0.0;
    f64 settle_max = 0.0;
    f64 never_settled = 0.0; // fraction of the changed tiles still outside the tolerance before the final frames
    std::vector<f32> settled_curve = {}; // per frame after start_frame: fraction of the changed tiles settled
};

// Result of a convergence measurement: per frame of the run, per metric (RT_LAB_MEASURE_METRIC_*), the tile histogram.
// cumulative[metric][frame][k] = tiles below bin k (under range + bins 0..k-1), k in [0, RT_LAB_MEASURE_BINS]; tiles
// above range or without a valid value never count as below. denominator[metric] = most tiles with geometry of any frame.
struct RtLabMeasurement
{
    i32 frames = 0;
    i32 window = 0;           // luma statistics window it was recorded with
    i32 preset = 0;           // RtLabPreset it was recorded with
    // log10 range of the histogram bins it was recorded with (results saved with older ranges stay readable).
    f32 log_min = RT_LAB_MEASURE_LOG_MIN;
    f32 log_max = RT_LAB_MEASURE_LOG_MAX;
    RtLabSettings settings = {}; // the preset's settings as recorded (split history forced on)
    f32 camera_position[3] = {};
    RtLabTestMove move = {};  // programmed move it was recorded with (move.enabled = 0: static camera)
    // Temporal lag (step response, rt_lab_measure_lag): per frame per RT_LAB_MEASURE_LAG_TILE^2 tile mean unexposed output
    // luminance (-1 = no geometry), frames x lag_tiles. Empty for results loaded from files (only `lag` is saved).
    i32 lag_tiles_x = 0;
    i32 lag_tiles_y = 0;
    i32 lag_start_frame = 0;  // the end of the programmed move (0 = static run: convergence from reset)
    std::vector<f32> lag_values = {};
    RtLabLagSummary lag = {};
    ProgrammedMovePose move_camera_origin = {}; // step 0 poses of the move (camera, sun)
    ProgrammedMovePose move_sun_origin = {};
    u32 resolution[2] = {};
    std::string timestamp = {};
    std::vector<u32> rows;    // frames x RT_LAB_MEASURE_ROW_SIZE, raw
    std::vector<u32> cumulative[RT_LAB_MEASURE_METRICS]; // frames x (RT_LAB_MEASURE_BINS + 1)
    u32 denominator[RT_LAB_MEASURE_METRICS] = {};
    u32 generation = 0;       // bumped on every new result (UI caches)
};

// Saved / unsaved measurement results, for comparing runs. Saved results live in RT_LAB_RESULTS_DIRECTORY as
// <name>.rtlab (rt_lab_result_save) and load lazily: scanning reads only the metadata header.
#define RT_LAB_RESULTS_DIRECTORY "rt_lab_results"
struct RtLabResult
{
    std::string name = {};
    std::filesystem::path path = {}; // empty = not saved yet
    bool loaded = false;             // measurement holds the histogram rows
    RtLabMeasurement measurement = {};
};

// Last convergence error estimate, copied from the general readback (custom - progressive, exposed irradiance).
struct RtLabConvergenceResult
{
    u32 pixels = 0;
    f64 mse = 0.0;
    f64 bias = 0.0;           // mean signed luminance difference
    f64 mean_abs_error = 0.0; // mean absolute luminance difference
    f32 histogram_range = 0.0f; // range the histogram was binned with
    std::array<u32, RT_LAB_CONVERGENCE_HISTOGRAM_BINS> histogram = {};
    u32 below_range = 0; // pixels left of the histogram
    u32 above_range = 0; // pixels right of the histogram
    // Split history error prediction check (Custom |A - B| / 2 vs actual |Custom - Progressive| luminance).
    u32 predicted_pixels = 0;
    f64 mean_predicted = 0.0;
    f64 mean_actual = 0.0;
    f64 rms_predicted = 0.0;
    f64 rms_actual = 0.0;
    f64 correlation = 0.0;
};

struct RtLabState
{
    // Presets. The active one drives the lab; both are editable in the UI.
    i32 active_preset = RT_LAB_PRESET_CUSTOM;
    std::array<RtLabSettings, RT_LAB_PRESET_COUNT> presets = {};

    // Effective sample count metric (RT_LAB_SPATIAL_SAMPLE_COUNT debug view), applied to rt_lab_settings every frame.
    f32 sample_count_input_clamp = 0.0f; // 0 = off
    f32 sample_count_debug_max = 4096.0f;

    // == Convergence error estimate ==
    // "Estimate Convergence Error": lock the camera, accumulate the progressive preset for test_frames frames and
    // capture the resolved diffuse, then the same for the custom preset, then compare the two every frame
    // (rt_lab_convergence.hlsl) until the next estimate.
    bool start_requested = false;
    bool cancel_requested = false;
    i32 test_frames = 512;
    f32 histogram_range = 0.25f; // histogram covers [-range, range] (manual)
    // Auto range: rescales over time (every analysis readback) towards the smallest symmetric range that holds
    // histogram_auto_coverage of the pixels. Starts at 4 x RMSE of the first result (manual range before that).
    bool histogram_auto_range = true;
    f32 histogram_auto_coverage = 0.995f;
    f32 histogram_auto_range_value = 0.0f; // current auto range, 0 = not initialized
    RtLabConvergencePhase phase = RtLabConvergencePhase::IDLE;
    i32 phase_frame = 0;
    bool camera_locked = false; // read by the application, skips camera input
    bool has_result = false;    // both captures are valid: the analysis pass runs
    u32 capture_target = 0;     // RT_LAB_CAPTURE_* this frame
    i32 saved_lab_enabled = 0;
    i32 saved_rtgi_enabled = 0;
    // Full res captures of the resolved diffuse. External: they must survive the graph rebuilds of the preset switches.
    daxa::ExternalTaskImage progressive_image = daxa::ExternalTaskImage{{.name = "rt_lab_convergence_progressive"}};
    daxa::ExternalTaskImage custom_image = daxa::ExternalTaskImage{{.name = "rt_lab_convergence_custom"}};
    daxa::ExternalTaskImage predicted_error_image = daxa::ExternalTaskImage{{.name = "rt_lab_convergence_predicted_error"}};
    RtLabConvergenceResult result = {};
    // Programmed move variant (estimate_move.enabled, see RtLabConvergencePhase::MOVE_*).
    RtLabTestMove estimate_move = {};
    i32 move_reference_frames = 128; // progressive frames per step
    i32 move_warmup_frames = 256;    // custom frames at step 0 before the move
    i32 move_step = 0;               // current step of the running estimate
    i32 move_steps = 0;              // last step (programmed_move_step_count of the longer move)
    i32 move_collect_countdown = 0;
    u32 analyze_step = ~0u;          // step the analysis pass compares this frame (~0u = none / static estimate)
    bool restore_custom_history = false; // this frame: custom history backups -> history (MOVE_CUSTOM frames)
    bool save_custom_history = false;    // this frame: custom history -> backups (every custom capture frame)
    bool backup_custom_history = false;  // graph built with the custom history backup tasks
    // Custom history backups (survive the graph rebuilds of the preset switches), by persistent image name.
    struct HistoryBackup
    {
        daxa::ExternalTaskImage image = {};
        daxa::Format format = {};
        daxa::Extent3D size = {};
    };
    std::map<std::string, HistoryBackup> history_backups = {};
    std::vector<RtLabConvergenceStep> move_results = {}; // per step of the last (running) move estimate

    // == Programmed move of a running test (read by the application) ==
    // test_move: the running test's move (null = none). The application poses the camera / sun at test_move_step every
    // frame and uses test_move->fixed_dt as the frame delta time; on test_move_begin it takes the origins first.
    RtLabTestMove const * test_move = nullptr;
    i32 test_move_step = 0;
    bool test_move_begin = false;
    ProgrammedMovePose test_camera_origin = {}; // step 0 poses of the running test (rt_lab_set_test_move_origins)
    ProgrammedMovePose test_sun_origin = {};
    // Preview of a test move (UI "Preview Move"): plays the move one step per frame without running the test, then the
    // poses return. Only while no test runs. preview_stop_requested ends it early.
    RtLabTestMove const * preview_move = nullptr;
    i32 preview_step = 0;
    bool preview_stop_requested = false;

    // == Convergence measurement ==
    // "Measure Convergence": lock the camera, reset the history and run the ACTIVE preset for measure_frames frames,
    // logging per frame tile histograms of the post-spatial convergence error and the output temporal instability
    // (rt_lab_measure.hlsl). Reference free. Analytics in the UI (rt_lab_measure_* helpers, rt_lab.hpp).
    bool measure_start_requested = false;
    bool measure_cancel_requested = false;
    i32 measure_frames = 512;
    i32 measure_window = 16; // luma statistics: average until this many frames, then an EMA over this many
    RtLabTestMove measure_move = {}; // one move step per measured frame, then the camera holds still
    RtLabMeasurePhase measure_phase = RtLabMeasurePhase::IDLE;
    i32 measure_frame = 0;
    i32 measure_collect_countdown = 0;
    u32 measure_row = ~0u;   // row the store pass writes this frame, ~0u = none
    daxa::ExternalTaskBuffer measure_frame_histogram = daxa::ExternalTaskBuffer{{.name = "rt_lab_measure_frame_histogram"}};
    daxa::ExternalTaskBuffer measure_run_histograms = daxa::ExternalTaskBuffer{{.name = "rt_lab_measure_run_histograms"}}; // host visible
    daxa::ExternalTaskBuffer measure_lag_run = daxa::ExternalTaskBuffer{{.name = "rt_lab_measure_lag_run"}}; // host visible, frames x lag tiles
    u64 measure_lag_capacity = 0; // floats
    f32 lag_tolerance = 0.02f;    // temporal lag tolerance (relative), UI
    i32 measure_run_capacity = 0; // rows measure_run_histograms holds
    RtLabMeasurement measurement = {}; // latest run
    std::vector<RtLabResult> results = {};
    i32 selected_result = -1;  // shown result (A)
    i32 compare_result = -1;   // optional comparison (B), -1 = none
    bool results_scanned = false;

    bool rebuild_requested = false; // the tests' state changes need a task graph rebuild (consumed by rt_lab_frame_update)

    // While RT Lab runs: PGI updates every probe every frame (pgi_settings.update_rate = PGI_UPDATE_RATE_FULL). The user's
    // rate is saved when the override starts and restored when it ends (unchecked or RT Lab off).
    bool force_full_pgi_update = false;

    // Post stabilization (inside the resolve, rt_lab_resolve.hlsl): TAA style variance clipped temporal step on the final
    // output. Not part of the presets (its settings go through the resolve's push constant). Graph shape: enabled.
    bool post_stabilization_enabled = true;
    f32 post_stabilization_alpha = 0.1f; // blend factor of the current output
    f32 post_stabilization_gamma = 1.0f; // neighborhood box half width in standard deviations
    i32 post_stabilization_radius = 1;   // variance estimator neighborhood (2r + 1)^2, 1..RT_LAB_STABILIZE_MAX_RADIUS
    bool post_stabilization_clamp = true; // false = no variance clamp (plain exponential history)
    bool post_stabilization_temporal_variance = true; // short (<= 4 frame) temporal input variance widens the clamp box
    bool post_stabilization_built = false; // graph built with the pass (rebuild trigger)
    bool pgi_update_rate_overridden = false;
    i32 saved_pgi_update_rate = 0;
};
