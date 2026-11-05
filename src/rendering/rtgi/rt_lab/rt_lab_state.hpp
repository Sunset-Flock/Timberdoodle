#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <daxa/utils/task_graph.hpp>

#include "../../../timberdoodle.hpp"
#include "../../../programmed_move.hpp"
#include "../../../shader_shared/readback.inl"
#include "rt_lab_settings.inl"
#include "rt_lab_measure_layout.inl"
#include "neural/rt_lab_neural_state.hpp"
#include "rt_lab_analysis.hpp"
#include "rt_lab_probe.hpp"
#include "rt_lab_telemetry.hpp"
#include "rt_lab2_state.hpp"
#include "rt_lab3_state.hpp"
#include "rt_lab4_state.hpp"
#include "rt_lab_integrator_state.hpp"

using namespace tido::types;

// CPU side RT Lab state, owned by RenderContext (render_context.rt_lab). rt_lab_frame_update (rt_lab.hpp) applies it
// to RenderGlobalData::rt_lab_settings at the start of every frame.

enum RtLabPreset : i32
{
    RT_LAB_PRESET_PROGRESSIVE = 0, // reference: full res, fp32 long accumulation, 2 tight a-trous passes
    RT_LAB_PRESET_CUSTOM = 1,      // the settings under test
    RT_LAB_PRESET_NEURAL = 2,      // the custom defaults with the neural affinity a-trous (RtLabState::neural, enabled while this preset is active)
    RT_LAB_PRESET_COUNT = 3,
};

// Programmed move of a test: camera and sun move, stepped with a FIXED time step (also the frame delta time while the
// test runs) so runs are reproducible. Step k pose = programmed_move_pose_at(origin, k).
struct RtLabTestMove
{
    bool enabled = false;
    f32 fixed_dt = 1.0f / 60.0f;
    ProgrammedMove camera = {.duration = 2.0f};
    ProgrammedMove sun = {.translation = false, .duration = 2.0f};
};

// Metrics of a test (RtLabTestSetup::metrics, RtLabMeasurement::metrics).
#define RT_LAB_TEST_METRIC_CONVERGENCE 0x1u // post-spatial split relative error (reference free)
#define RT_LAB_TEST_METRIC_INSTABILITY 0x2u // temporal instability of the output
#define RT_LAB_TEST_METRIC_LAG 0x4u         // temporal lag (step response)
#define RT_LAB_TEST_METRIC_REFERENCE 0x8u   // error vs the progressive preset reference
#define RT_LAB_TEST_METRIC_JITTER 0x10u     // frame to frame jitter of the output (second temporal difference)
#define RT_LAB_TEST_METRIC_HISTOGRAMS (RT_LAB_TEST_METRIC_CONVERGENCE | RT_LAB_TEST_METRIC_INSTABILITY | RT_LAB_TEST_METRIC_JITTER)
#define RT_LAB_TEST_METRIC_PER_FRAME (RT_LAB_TEST_METRIC_HISTOGRAMS | RT_LAB_TEST_METRIC_LAG)
#define RT_LAB_TEST_METRIC_DEFAULT RT_LAB_TEST_METRIC_PER_FRAME

// The RT Lab test: one setup, the checked metrics decide what runs. The tested preset is the ACTIVE one.
//   reference (static only): the progressive preset accumulates reference_frames frames from a reset, last one captured
//   run (always):            the tested preset accumulates `frames` frames from a reset (one move step per frame, then
//                            the poses hold), logging the checked per frame metrics. Static reference: last frame
//                            captured and compared with the reference
//   reference (move):        after the run, the tested preset warms up at step 0, then per move step: the tested preset
//                            reprojects + adds ONE frame (captured), the progressive preset accumulates
//                            move_reference_frames frames from a reset at the same pose (captured + compared)
struct RtLabTestSetup
{
    i32 frames = 512;
    i32 window = 16; // instability: average until this many frames, then an EMA over this many
    RtLabTestMove move = {};
    u32 metrics = RT_LAB_TEST_METRIC_DEFAULT; // RT_LAB_TEST_METRIC_*
    i32 reference_frames = 512;      // static reference: progressive frames
    i32 move_reference_frames = 128; // move reference: progressive frames per step (made even at the start)
    i32 move_warmup_frames = 256;    // move reference: tested frames at step 0 before the move
    // Temporal shock: the point light culling (light_settings.cull_all_point_lights) is flipped for run frames
    // [shock_start, shock_start + shock_frames) (1 frame = single frame flash), the lag metric starts at its end.
    // 0 frames = off.
    i32 shock_start = 0;
    i32 shock_frames = 0;
};

enum struct RtLabTestPhase
{
    IDLE,
    REFERENCE,      // static reference: progressive accumulation
    RUN,            // tested preset, one measurement row per frame
    MOVE_WARMUP,    // move reference: tested preset at step 0 from a reset, captured on its last frame
    MOVE_TESTED,    // move reference: tested preset, one frame at the current step (history restored from the backups)
    MOVE_REFERENCE, // move reference: progressive at the current step, captured + compared on its last frame
    COLLECTING,     // waiting for the GPU (measurement rows, last analysis readback)
};

// Reference comparison (exposed irradiance, tested - progressive), from the convergence analysis readback.
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
    // Split history error prediction check (tested |A - B| / 2 vs actual |tested - progressive| luminance).
    u32 predicted_pixels = 0;
    f64 mean_predicted = 0.0;
    f64 mean_actual = 0.0;
    f64 rms_predicted = 0.0;
    f64 rms_actual = 0.0;
    f64 correlation = 0.0;
};

// Reference comparison of one programmed move step (tested after one reprojected frame vs a fresh progressive reference
// at the same pose).
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
    std::vector<f32> mean_curve = {};    // per frame of the run: mean unexposed output luma over the tiles (not saved in .rtlab)
};

// Result of a test: per frame of the run, per metric (RT_LAB_MEASURE_METRIC_*), the tile histogram, the lag data and the
// reference comparison. cumulative[metric][frame][k] = tiles below bin k (under range + bins 0..k-1), k in
// [0, RT_LAB_MEASURE_BINS]; tiles above range or without a valid value never count as below. denominator[metric] = most
// tiles with geometry of any frame.
struct RtLabMeasurement
{
    i32 frames = 0;
    i32 window = 0;           // luma statistics window it was recorded with
    i32 preset = 0;           // RtLabPreset it was recorded with
    // RT_LAB_TEST_METRIC_* recorded (results saved before the test setup: the per frame ones). Unrecorded metrics have
    // empty histograms / no lag data.
    u32 metrics = RT_LAB_TEST_METRIC_PER_FRAME;
    // log10 range of the histogram bins it was recorded with (results saved with older ranges stay readable).
    f32 log_min = RT_LAB_MEASURE_LOG_MIN;
    f32 log_max = RT_LAB_MEASURE_LOG_MAX;
    RtLabSettings settings = {}; // the preset's settings as recorded (split history forced on)
    f32 camera_position[3] = {};
    RtLabTestMove move = {};  // programmed move it was recorded with (move.enabled = 0: static camera)
    ProgrammedMovePose move_camera_origin = {}; // step 0 poses of the move (camera, sun)
    ProgrammedMovePose move_sun_origin = {};
    // Temporal lag (step response, rt_lab_measure_lag): per frame per RT_LAB_MEASURE_LAG_TILE^2 tile mean unexposed output
    // luminance (-1 = no geometry), frames x lag_tiles. Empty for results loaded from files (only `lag` is saved).
    i32 lag_tiles_x = 0;
    i32 lag_tiles_y = 0;
    i32 lag_start_frame = 0;  // the end of the programmed move (0 = static run: convergence from reset)
    std::vector<f32> lag_values = {};
    RtLabLagSummary lag = {};
    // Reference comparison (RT_LAB_TEST_METRIC_REFERENCE): static = the run's last frame vs the progressive reference;
    // programmed move = per step errors (reference_steps), reference = the last step.
    RtLabConvergenceResult reference = {};
    i32 reference_frames = 0; // static: progressive frames; move: progressive frames per step
    i32 reference_warmup_frames = 0; // move: tested frames at step 0
    std::vector<RtLabConvergenceStep> reference_steps = {};
    u32 resolution[2] = {};
    std::string timestamp = {};
    std::vector<u32> rows;    // frames x RT_LAB_MEASURE_ROW_SIZE, raw
    std::vector<u32> cumulative[RT_LAB_MEASURE_METRICS]; // frames x (RT_LAB_MEASURE_BINS + 1)
    u32 denominator[RT_LAB_MEASURE_METRICS] = {};
    u32 generation = 0;       // bumped on every new result (UI caches)
};

// Saved / unsaved test results, for comparing runs. Saved results live in RT_LAB_RESULTS_DIRECTORY as <name>.rtlab
// (rt_lab_result_save) and load lazily: scanning reads only the metadata header.
#define RT_LAB_RESULTS_DIRECTORY "rt_lab_results"
struct RtLabResult
{
    std::string name = {};
    std::filesystem::path path = {}; // empty = not saved yet
    bool loaded = false;             // measurement holds the histogram rows
    RtLabMeasurement measurement = {};
};

// == Pipeline ==
// Which stages run is part of the preset (RtLabSettings::restir_enabled / denoiser). The RT Lab always runs
//   reproject + ray redistribution -> trace -> [ReSTIR] -> ONE denoiser (classic temporal + spatial, OR the hash grid) -> temporal resolve -> resolve
// The four lab versions are combinations (rt_lab_pipeline_set_version). The stage implementations are still the per version task graphs (tasks_rt_lab_main,
// tasks_rt_lab2_main, tasks_rt_lab3_main, tasks_rt_lab4_main, tasks_rt_lab_restir_main); rt_lab_pipeline_resolve derives which one runs from the active preset.

// 1 = classic, 2 = hash grid, 3 = ReSTIR only, 4 = ReSTIR -> hash grid; 0 = another combination (ReSTIR -> classic).
inline auto rt_lab_pipeline_version(RtLabSettings const & s) -> i32
{
    if (s.restir_enabled == 0)
    {
        return s.denoiser == RT_LAB_DENOISER_CLASSIC ? 1 : (s.denoiser == RT_LAB_DENOISER_HASH_GRID ? 2 : 0);
    }
    return s.denoiser == RT_LAB_DENOISER_NONE ? 3 : (s.denoiser == RT_LAB_DENOISER_HASH_GRID ? 4 : 0);
}

inline void rt_lab_pipeline_set_version(RtLabSettings & s, i32 version)
{
    switch (version)
    {
        case 1: s.restir_enabled = 0; s.denoiser = RT_LAB_DENOISER_CLASSIC; break;
        case 2: s.restir_enabled = 0; s.denoiser = RT_LAB_DENOISER_HASH_GRID; break;
        case 3: s.restir_enabled = 1; s.denoiser = RT_LAB_DENOISER_NONE; break;
        case 4: s.restir_enabled = 1; s.denoiser = RT_LAB_DENOISER_HASH_GRID; break;
        default: break;
    }
}

// Everything of the lab that is not in RtLabSettings (the GPU shared part of a preset) but belongs to a preset: the settings of the ReSTIR stage, the hash grid,
// the glue between them and the denoiser behind, and the temporal resolve. RtLabState::preset_configs[preset]; the live states (lab2 / lab3 / lab4 / integrator)
// hold the active preset's copy (rt_lab_config_apply every frame), the UI edits the preset's.
struct RtLabPresetConfig
{
    RtLab2Config lab2 = {};
    RtLab3Config lab3 = {};
    RtLab4Config lab4 = {};
    RtLabIntegratorConfig integrator = {};
};

struct RtLabState
{
    // The pipeline of the active preset as the task graph was last built for it (rt_lab_pipeline_resolve). The "enabled" flags of lab2 / lab3 / lab4 are derived.
    i32 built_pipeline = -1;
    // Presets. The active one drives the lab (and is the tested one); both are editable in the UI.
    i32 active_preset = RT_LAB_PRESET_CUSTOM;
    std::array<RtLabSettings, RT_LAB_PRESET_COUNT> presets = {};
    std::array<RtLabPresetConfig, RT_LAB_PRESET_COUNT> preset_configs = {};

    // Effective sample count metric (RT_LAB_SPATIAL_SAMPLE_COUNT debug view), applied to rt_lab_settings every frame.
    f32 sample_count_input_clamp = 0.0f; // 0 = off
    f32 sample_count_debug_max = 4096.0f;

    // == Test (RtLabTestSetup) ==
    RtLabTestSetup test = {};          // edited by the UI / automation, copied into `running` at the start
    bool test_start_requested = false;
    bool test_cancel_requested = false;
    RtLabTestSetup running = {};       // setup of the running (last) test
    RtLabTestPhase phase = RtLabTestPhase::IDLE;
    i32 phase_frame = 0;
    i32 move_step = 0;                 // move reference: current step
    i32 move_steps = 0;                // move reference: last step (rt_lab_test_move_steps)
    i32 collect_countdown = 0;
    bool camera_locked = false;        // read by the application, skips camera input
    i32 saved_lab_enabled = 0;
    i32 saved_rtgi_enabled = 0;
    RtLabMeasurement measurement = {}; // running / latest test
    // Per frame passes (rt_lab_frame_update): measurement row the store pass writes (~0u = none), capture target
    // (RT_LAB_CAPTURE_*), step the analysis pass compares (~0u = none / static reference).
    u32 measure_row = ~0u;
    u32 capture_target = 0;
    u32 analyze_step = ~0u;
    daxa::ExternalTaskBuffer measure_frame_histogram = daxa::ExternalTaskBuffer{{.name = "rt_lab_measure_frame_histogram"}};
    daxa::ExternalTaskBuffer measure_run_histograms = daxa::ExternalTaskBuffer{{.name = "rt_lab_measure_run_histograms"}}; // host visible
    daxa::ExternalTaskBuffer measure_lag_run = daxa::ExternalTaskBuffer{{.name = "rt_lab_measure_lag_run"}}; // host visible, frames x lag tiles
    u64 measure_lag_capacity = 0; // floats
    i32 measure_run_capacity = 0; // rows measure_run_histograms holds
    f32 lag_tolerance = 0.02f;    // temporal lag tolerance (relative), UI

    // == Reference comparison ==
    // Full res captures of the resolved diffuse. External: they must survive the graph rebuilds of the preset switches.
    // Once a static reference compared them, the analysis pass keeps running every frame (live_analysis: debug views, the
    // live histogram) until the next test.
    daxa::ExternalTaskImage progressive_image = daxa::ExternalTaskImage{{.name = "rt_lab_convergence_progressive"}};
    daxa::ExternalTaskImage custom_image = daxa::ExternalTaskImage{{.name = "rt_lab_convergence_custom"}};
    daxa::ExternalTaskImage predicted_error_image = daxa::ExternalTaskImage{{.name = "rt_lab_convergence_predicted_error"}};
    bool live_analysis = false;
    RtLabConvergenceResult live_result = {}; // latest analysis readback
    f32 histogram_range = 0.25f; // histogram covers [-range, range] (manual)
    // Auto range: rescales over time (every analysis readback) towards the smallest symmetric range that holds
    // histogram_auto_coverage of the pixels. Starts at 4 x RMSE of the first result (manual range before that).
    bool histogram_auto_range = true;
    f32 histogram_auto_coverage = 0.995f;
    f32 histogram_auto_range_value = 0.0f; // current auto range, 0 = not initialized
    // Move reference: the tested history survives the progressive phases' graph rebuilds through backup copies.
    bool restore_custom_history = false; // this frame: history backups -> history (MOVE_TESTED frames)
    bool save_custom_history = false;    // this frame: history -> backups (every tested capture frame)
    bool backup_custom_history = false;  // graph built with the history backup tasks
    struct HistoryBackup
    {
        daxa::ExternalTaskImage image = {};
        daxa::Format format = {};
        daxa::Extent3D size = {};
    };
    std::map<std::string, HistoryBackup> history_backups = {}; // by persistent image name

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

    // == Results ==
    std::vector<RtLabResult> results = {};
    i32 selected_result = -1;  // shown result (A)
    i32 compare_result = -1;   // optional comparison (B), -1 = none
    bool results_scanned = false;

    bool rebuild_requested = false; // the tests' state changes need a task graph rebuild (consumed by rt_lab_frame_update)

    // While RT Lab runs: PGI updates every probe every frame (pgi_settings.update_rate = PGI_UPDATE_RATE_FULL). The user's
    // rate is saved when the override starts and restored when it ends (unchecked or RT Lab off).
    bool force_full_pgi_update = false;
    // Temporal shock of a running test (RtLabTestSetup::shock_*): the culling flag before the test, restored after it.
    bool shock_active = false;
    bool shock_saved_cull_point_lights = true;

    // Pixel telemetry (rt_lab_telemetry.hpp): rays + surface detail of the pixel under the held mouse button.
    RtLabTelemetryState telemetry = {};
    // Live pixel probes (rt_lab_probe.hpp): the accumulated denoiser state of two followed pixels.
    RtLabProbeState probe = {};
    // Analysis on the resolved output (rt_lab_analysis.hpp): statistics + pixel picker.
    RtLabAnalysisState analysis = {};

    // Neural affinity (neural/rt_lab_neural.hpp): learned similarity of the a-trous taps, training data dump.
    RtLabNeuralState neural = {};
    // RT Lab 2 (rt_lab2.hpp): the hash grid denoiser, an alternative to everything above that denoises (the trace and the tests are shared).
    RtLab2State lab2 = {};
    // RT Lab 3 (rt_lab3.hpp): ReSTIR GI on the same trace, an alternative to the denoisers above.
    RtLab3State lab3 = {};
    // RT Lab 4 (rt_lab4.hpp): the ReSTIR stage of lab 3 feeding the cells of lab 2.
    RtLab4State lab4 = {};
    // Temporal resolve of the original lab (rt_lab_integrator.hpp), runs on the final lab image right before the resolve. Not part of the presets.
    RtLabIntegratorState integrator = {};
    bool pgi_update_rate_overridden = false;
    i32 saved_pgi_update_rate = 0;
};

inline auto rt_lab_test_running(RtLabState const & lab) -> bool
{
    return lab.phase != RtLabTestPhase::IDLE;
}

// The live config (what the passes read) as a preset config / the other way round.
inline auto rt_lab_config_gather(RtLabState const & lab) -> RtLabPresetConfig
{
    return RtLabPresetConfig{lab.lab2, lab.lab3, lab.lab4, lab.integrator};
}

inline void rt_lab_config_apply(RtLabState & lab, RtLabPresetConfig const & config)
{
    static_cast<RtLab2Config &>(lab.lab2) = config.lab2;
    static_cast<RtLab3Config &>(lab.lab3) = config.lab3;
    static_cast<RtLab4Config &>(lab.lab4) = config.lab4;
    static_cast<RtLabIntegratorConfig &>(lab.integrator) = config.integrator;
}
