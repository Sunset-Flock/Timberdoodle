#pragma once

// Standart headers:
#include <chrono>
#include <memory>
// Library headers:
// Project headers:
#include "timberdoodle.hpp"
using namespace tido::types;

#include "window.hpp"

#include "scene/scene.hpp"
#include "scene/asset_processor.hpp"
#include "ui/ui.hpp"
#include "rendering/renderer.hpp"
#include "gpu_context.hpp"
#include "multithreading/thread_pool.hpp"
#include "application_state.hpp"
#include <nlohmann/json.hpp>

// Automated performance test: teleport the camera, load a scene, wait until all assets are
// streamed in, then for each view: teleport, render wait_frames frames, dump the smoothed GPU
// timings + a screenshot into output_dir. Exits when all views are done.
struct PerfTestView
{
    f32vec3 position = {};
    f32 yaw = {};
    f32 pitch = {};
};

struct PerfTestInfo
{
    std::filesystem::path scene_path = {};
    // Measured one after another after a single scene load. Empty = measure the current camera.
    // With more than one view, outputs are named <name>_view<i>_*.
    std::vector<PerfTestView> views = {};
    u32 wait_frames = 1000;
    std::filesystem::path output_dir = "perf_tests";
    std::string name = "perf";
    // > 0: start an RT Lab convergence error estimate with this many frames per preset when each view starts measuring
    // (--frames must cover 2x this). The result is written into the timings json.
    i32 rt_lab_estimate_frames = 0;
    // > 0: start an RT Lab convergence measurement of the active preset with this many frames when each view starts
    // measuring (--frames must cover it + a few frames). Summary written into the timings json.
    i32 rt_lab_measure_frames = 0;
    // Programmed move of both RT Lab tests (--rt-lab-move / --rt-lab-sun-move / --rt-lab-move-dt), enabled when set.
    RtLabTestMove rt_lab_move = {};
    i32 rt_lab_move_warmup_frames = 256;    // move estimate: custom frames at step 0
    i32 rt_lab_move_reference_frames = 128; // move estimate: progressive frames per step
    bool rt_lab_move_sun_set = false;       // --rt-lab-sun-move given (else --rt-lab-move zeroes the sun move)
};

// Test plan (--test-plan <file.json>, application_test_plan.cpp): one scene load, then every test of the plan in order,
// results appended to <out>/<name>_results.json after each test. Format: see application_test_plan.cpp.
enum struct TestPlanTestType
{
    PERF,     // render `frames` frames, then write the GPU timings
    MEASURE,  // RT Lab convergence measurement of the active preset, `frames` frames
    ESTIMATE, // RT Lab convergence error vs progressive (`frames` per preset; with a move: move_warmup / move_reference)
};

struct TestPlanEntry
{
    std::string name = {};
    nlohmann::json spec = {}; // the entry as given (echoed into the results)
    TestPlanTestType type = TestPlanTestType::MEASURE;
    i32 frames = 256;
    i32 settle_frames = 16;   // frames rendered after the setup before the test starts (graph rebuild, PGI, exposure)
    bool has_camera = false;
    ProgrammedMovePose camera = {};
    bool has_sun = false;
    ProgrammedMovePose sun = {}; // yaw = Angle X, pitch = Angle Y
    i32 debug_draw = 0;
    i32 active_preset = RT_LAB_PRESET_CUSTOM;
    std::array<RtLabSettings, RT_LAB_PRESET_COUNT> presets = {};
    RtLabTestMove move = {};
    i32 move_warmup_frames = 256;
    i32 move_reference_frames = 128;
    bool screenshot = false;
    bool force_full_pgi_update = false;
    bool post_stabilization_enabled = true;
    f32 post_stabilization_alpha = 0.1f;
    f32 post_stabilization_gamma = 1.0f;
    i32 post_stabilization_radius = 1;
    bool post_stabilization_clamp = true;
    bool post_stabilization_temporal_variance = true;
};

struct Application
{
public:
    Application(i32vec2 window_size = {1024, 1024});
    ~Application();

    auto run() -> i32;
    auto load_scene(std::filesystem::path const & path) -> bool;
    void set_camera(f32vec3 position, f32 yaw, f32 pitch);
    // Selects a DEBUG_DRAW_MODE_* (shader_shared/shared.inl) as if picked in the main debug dropdown.
    void set_debug_draw_mode(i32 mode);
    void set_rtgi_specular_enabled(bool enabled);
    void set_upward_gloss(f32 gloss);
    // Selects the RT Lab denoiser instead of the RTGI denoiser.
    void set_rt_lab(bool half_res, bool progressive, bool no_spatial, bool atrous);
    // RT Lab custom preset temporal gradient sample selection (RtLabSettings::temporal_gradient_ris).
    void set_rt_lab_gradient_ris(i32 ris);
    // RT Lab custom preset temporal gradient on / off (RtLabSettings::temporal_gradient_enabled).
    void set_rt_lab_gradient(i32 enabled);
    // RT Lab custom preset coarse temporal anti-lag on / off (RtLabSettings::coarse_anti_lag_enabled).
    void set_rt_lab_coarse_anti_lag(i32 enabled);
    void start_perf_test(PerfTestInfo const & info);
    // Parses the plan and starts it (scene load first). false = plan unreadable / invalid (message printed).
    auto start_test_plan(std::filesystem::path const & path) -> bool;

private:
    void update();
    void update_perf_test();
    // Camera / sun programmed moves: the RT Lab tests' fixed step moves (+ fixed delta time) or the property viewer's
    // free runs. Called in update() before the camera input.
    void update_programmed_moves();
    void start_perf_test_rt_lab_estimate();
    void write_perf_test_timings(std::filesystem::path const & path);
    // The perf test result blocks of the current state: camera, resolution, RT Lab estimate / measurement, GPU timings.
    auto perf_test_result_json() -> nlohmann::json;
    void update_test_plan();
    void test_plan_setup(TestPlanEntry const & entry);
    void test_plan_finish_test(std::string const & error);
    void test_plan_next();

    enum struct TestPlanPhase
    {
        NONE,
        LOADING,    // scene streaming in
        SETTLE,     // test set up, rendering settle frames
        RUNNING,    // test running
        SCREENSHOT, // waiting for the screenshot write
    };
    TestPlanPhase _test_plan_phase = TestPlanPhase::NONE;
    std::filesystem::path _test_plan_path = {};
    std::filesystem::path _test_plan_scene = {};
    std::filesystem::path _test_plan_output_dir = "perf_tests";
    std::string _test_plan_name = "plan";
    std::vector<TestPlanEntry> _test_plan_entries = {};
    nlohmann::json _test_plan_results = {};
    u32 _test_plan_index = 0;
    i32 _test_plan_frame_counter = 0;
    bool _test_plan_test_started = false; // the RT Lab test was seen running
    std::chrono::steady_clock::time_point _test_plan_test_start_time = {};

    enum struct PerfTestPhase
    {
        NONE,
        LOADING,
        WARMUP,
        WRITING_SCREENSHOT,
    };
    // Loading counts as done once the asset pool is idle, nothing got uploaded and no BLAS
    // build is queued for this many frames in a row.
    static constexpr u32 PERF_TEST_SETTLE_FRAMES = 8;
    PerfTestInfo _perf_test = {};
    PerfTestPhase _perf_test_phase = PerfTestPhase::NONE;
    u32 _perf_test_frame_counter = 0;
    u32 _perf_test_view_index = 0;
    std::shared_ptr<Task> _perf_test_screenshot_task = {};
    i32 _exit_code = 0;
    bool _uploaded_assets_this_frame = false;
    // Camera / sun poses from before an RT Lab test's programmed move, restored when the test ends.
    bool _test_move_restore_pending = false;
    ProgrammedMovePose _test_move_restore_camera = {};
    ProgrammedMovePose _test_move_restore_sun = {};
    /**
        * EXPLANATION: Why do we use unique pointers here?
        * Many of these members are non-movable.
        * They can NOT be made movable easily!
        * They require dependency injection between each other!
        * A pattern that solves these problems (non-movable + dep injection), is to wrap these structs in heap allocations and refer to them only with pointers.
        * We do NOT need shared_ptr here, as we know the lifetime of these members beforehand! The Application controls their lifetime!
        * If member B referes to member A, it will be below it in the struct delaration. This means it will be destroyed before A. This way we wont have dangling pointers.
        * There are no performance implications of this as these structs are VERY low frequency creation/deletion and are never copied.
        * This allows the construction of Application to be much simpler, less bug prone and makes Application movable!
        * A good rule of thumb is to have structs movable. If you need members that are not movable, simply wrap them in pointers.
        * WARNING: THIS CAN ONLY BE APPLIED LIKE THIS FOR LOW FREQUENCY TYPES, AS IT MIGHT INCUR PERFORMANCE PROBLEMS OTHERWISE!
        */
    std::unique_ptr<Window> _window = {};
    std::unique_ptr<GPUContext> _gpu_context = {};
    std::unique_ptr<Scene> _scene = {};
    std::unique_ptr<AssetProcessor> _asset_manager = {};
    std::unique_ptr<UIEngine> _ui_engine = {};
    std::unique_ptr<Renderer> _renderer = {};
    std::unique_ptr<ThreadPool> _threadpool = {};
    std::vector<AssetProcessor::MeshLodGroupUploadInfo> _pending_mesh_uploads = {};
    ApplicationState app_state = {};
};