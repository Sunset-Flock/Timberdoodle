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
    void start_perf_test(PerfTestInfo const & info);

private:
    void update();
    void update_perf_test();
    void write_perf_test_timings(std::filesystem::path const & path);

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