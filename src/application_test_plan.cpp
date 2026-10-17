#include "application.hpp"
#include "rendering/rtgi/rt_lab/rt_lab.hpp"

#include <fmt/format.h>
#include <fstream>

// Test plan (--test-plan <file.json>): the scene is loaded ONCE, then every test runs in order without restarting the
// app. Results are appended to <out>/<name>_results.json after every test (a crash keeps the finished ones).
//
// {
//   "scene": "assets/bistro_lights_compressed/bistro_lights_c.gltf",   required
//   "resolution": [1280, 720],                                           optional (window size, default 1024 1024)
//   "out": "perf_tests/plan",                                            optional (default perf_tests)
//   "name": "combo",                                                     optional (default the plan file name)
//   "defaults": { ...test fields... },                                   optional, merged under every test (objects merge)
//   "tests": [
//     {
//       "name": "g1c1",                                     default test<index>
//       "test": "measure" | "estimate" | "perf",            default measure
//       "frames": 200,                                      measure: frames; estimate: frames per preset (static);
//                                                           perf: frames rendered before the timings are taken
//       "settle_frames": 16,                                frames rendered after the setup, before the test starts
//       "camera": "px py pz fx fy fz" | [x, y, z, yaw, pitch],   View Camera "Copy to clipboard" format or yaw / pitch
//       "sun": "angle_x angle_y" | [angle_x, angle_y],      sun panel angles (degrees)
//       "debug_draw": 0,                                    DEBUG_DRAW_MODE_* (int)
//       "preset": "custom" | "progressive",                 active RT Lab preset (measure / perf), default custom
//       "force_full_pgi_update": false,                     RT Lab "Force Full PGI Update" for this test
//       "post_stabilization": { "enabled": true, "alpha": 0.1, "gamma": 1.0, "radius": 1, "clamp": true, "temporal_variance": true },   RT Lab post stabilization (default on)
//       "custom": { "coarse_anti_lag_enabled": 1, ... },       RT Lab preset overrides, keys = saved result settings keys,
//       "progressive": { ... },                             applied to the app's presets as they were at plan start
//       "move": {                                           programmed move of the test (both test types)
//         "dt": 0.0166667,                                  time per step
//         "camera": { "velocity": [vx, vy, vz], "yaw_speed": 0, "pitch_speed": 0, "duration": 1, "start": "px py pz fx fy fz" },
//         "sun": { "yaw_speed": -25, "pitch_speed": -35, "duration": 2, "start": "250 90" }
//       },                                                  a missing camera / sun move = none (duration 0)
//       "move_frames": [256, 128],                          move estimate: custom warmup, progressive frames per step
//       "screenshot": false                                 <out>/<name>_<test>_screenshot.png after the test
//     }
//   ]
// }
// Per test result: the perf test blocks (camera, rt_lab_measure / rt_lab_convergence (+ move_steps), groups = GPU
// timings) + "name", "spec" (the merged entry), "wall_seconds", "error" when the test failed.

static auto test_plan_type(std::string const & type, TestPlanTestType & out) -> bool
{
    if (type == "perf")
    {
        out = TestPlanTestType::PERF;
    }
    else if (type == "measure")
    {
        out = TestPlanTestType::MEASURE;
    }
    else if (type == "estimate")
    {
        out = TestPlanTestType::ESTIMATE;
    }
    else
    {
        return false;
    }
    return true;
}

// Camera: "px py pz fx fy fz" or [x, y, z, yaw, pitch]. Sun: "angle_x angle_y" or [angle_x, angle_y].
static auto test_plan_pose(nlohmann::json const & json, bool camera, ProgrammedMovePose & pose) -> bool
{
    if (json.is_string())
    {
        return programmed_move_pose_from_text(json.get<std::string>().c_str(), camera, pose);
    }
    if (camera && json.is_array() && json.size() == 5)
    {
        pose = ProgrammedMovePose{
            .position = {json[0].get<f32>(), json[1].get<f32>(), json[2].get<f32>()},
            .yaw = json[3].get<f32>(),
            .pitch = json[4].get<f32>(),
        };
        return true;
    }
    if (!camera && json.is_array() && json.size() == 2)
    {
        pose = ProgrammedMovePose{.yaw = json[0].get<f32>(), .pitch = json[1].get<f32>()};
        return true;
    }
    return false;
}

auto Application::start_test_plan(std::filesystem::path const & path) -> bool
{
    std::ifstream file(path);
    nlohmann::json const plan = nlohmann::json::parse(file, nullptr, false);
    if (!file || plan.is_discarded() || !plan.is_object() || !plan.contains("scene") || !plan.contains("tests") || !plan["tests"].is_array())
    {
        fmt::print("[TestPlan] ERROR: \"{}\" is not a test plan (needs \"scene\" and a \"tests\" array)\n", path.string());
        return false;
    }
    auto & lab = _renderer->render_context->rt_lab;
    _test_plan_path = path;
    _test_plan_scene = plan["scene"].get<std::string>();
    _test_plan_output_dir = plan.value("out", std::string("perf_tests"));
    _test_plan_name = plan.value("name", path.stem().string());
    nlohmann::json const defaults = plan.value("defaults", nlohmann::json::object());
    _test_plan_entries.clear();
    for (u32 i = 0; i < plan["tests"].size(); ++i)
    {
        nlohmann::json spec = defaults;
        spec.update(plan["tests"][i], true);
        TestPlanEntry entry = {};
        entry.name = spec.value("name", fmt::format("test{}", i));
        entry.spec = spec;
        auto const fail = [&](std::string const & message) -> bool
        {
            fmt::print("[TestPlan] ERROR in test {} \"{}\": {}\n", i, entry.name, message);
            return false;
        };
        if (!test_plan_type(spec.value("test", std::string("measure")), entry.type))
        {
            return fail("\"test\" must be measure, estimate or perf");
        }
        entry.frames = spec.value("frames", entry.frames);
        entry.settle_frames = spec.value("settle_frames", entry.settle_frames);
        if (spec.contains("camera"))
        {
            entry.has_camera = test_plan_pose(spec["camera"], true, entry.camera);
            if (!entry.has_camera)
            {
                return fail("\"camera\" must be \"px py pz fx fy fz\" or [x, y, z, yaw, pitch]");
            }
        }
        if (spec.contains("sun"))
        {
            entry.has_sun = test_plan_pose(spec["sun"], false, entry.sun);
            if (!entry.has_sun)
            {
                return fail("\"sun\" must be \"angle_x angle_y\" or [angle_x, angle_y]");
            }
        }
        entry.debug_draw = spec.value("debug_draw", 0);
        std::string const preset = spec.value("preset", std::string("custom"));
        if (preset != "custom" && preset != "progressive")
        {
            return fail("\"preset\" must be custom or progressive");
        }
        entry.active_preset = preset == "custom" ? RT_LAB_PRESET_CUSTOM : RT_LAB_PRESET_PROGRESSIVE;
        for (i32 p = 0; p < RT_LAB_PRESET_COUNT; ++p)
        {
            char const * const key = p == RT_LAB_PRESET_CUSTOM ? "custom" : "progressive";
            std::vector<std::string> unknown = {};
            entry.presets[p] = rt_lab_settings_apply_json(lab.presets[p], spec.value(key, nlohmann::json::object()), unknown);
            if (!unknown.empty())
            {
                return fail(fmt::format("unknown \"{}\" setting \"{}\"", key, unknown.front()));
            }
        }
        if (spec.contains("move"))
        {
            nlohmann::json const & move = spec["move"];
            entry.move.enabled = true;
            entry.move.fixed_dt = std::max(move.value("dt", 1.0f / 60.0f), 1e-4f);
            entry.move.camera = move.contains("camera") ? programmed_move_from_json(move["camera"], ProgrammedMove{.duration = 0.0f}) : ProgrammedMove{.duration = 0.0f};
            entry.move.sun = move.contains("sun") ? programmed_move_from_json(move["sun"], ProgrammedMove{.translation = false, .duration = 0.0f})
                                                  : ProgrammedMove{.translation = false, .duration = 0.0f};
            entry.move.camera.translation = true;
            entry.move.sun.translation = false;
        }
        if (spec.contains("move_frames") && spec["move_frames"].is_array() && spec["move_frames"].size() == 2)
        {
            entry.move_warmup_frames = spec["move_frames"][0].get<i32>();
            entry.move_reference_frames = spec["move_frames"][1].get<i32>();
        }
        entry.screenshot = spec.value("screenshot", false);
        entry.force_full_pgi_update = spec.value("force_full_pgi_update", false);
        if (spec.contains("post_stabilization"))
        {
            nlohmann::json const & stabilization = spec["post_stabilization"];
            entry.post_stabilization_enabled = stabilization.value("enabled", true);
            entry.post_stabilization_alpha = stabilization.value("alpha", entry.post_stabilization_alpha);
            entry.post_stabilization_gamma = stabilization.value("gamma", entry.post_stabilization_gamma);
            entry.post_stabilization_radius = stabilization.value("radius", entry.post_stabilization_radius);
            entry.post_stabilization_clamp = stabilization.value("clamp", entry.post_stabilization_clamp);
            entry.post_stabilization_temporal_variance = stabilization.value("temporal_variance", entry.post_stabilization_temporal_variance);
        }
        _test_plan_entries.push_back(std::move(entry));
    }
    if (_test_plan_entries.empty())
    {
        fmt::print("[TestPlan] ERROR: \"{}\" has no tests\n", path.string());
        return false;
    }
    _test_plan_results = {
        {"plan", path.string()},
        {"name", _test_plan_name},
        {"scene", _test_plan_scene.string()},
        {"tests", nlohmann::json::array()},
    };
    if (_test_plan_entries[0].has_camera)
    {
        set_camera(_test_plan_entries[0].camera.position, _test_plan_entries[0].camera.yaw, _test_plan_entries[0].camera.pitch);
    }
    fmt::print("[TestPlan] \"{}\": {} tests, loading scene {}\n", _test_plan_name, _test_plan_entries.size(), _test_plan_scene.string());
    if (!load_scene(_test_plan_scene))
    {
        fmt::print("[TestPlan] ERROR: failed to load scene {}\n", _test_plan_scene.string());
        return false;
    }
    _test_plan_phase = TestPlanPhase::LOADING;
    _test_plan_frame_counter = 0;
    _test_plan_index = 0;
    return true;
}

void Application::test_plan_setup(TestPlanEntry const & entry)
{
    auto & render_context = *_renderer->render_context;
    auto & lab = render_context.rt_lab;
    fmt::print("[TestPlan] test {}/{} \"{}\"\n", _test_plan_index + 1, _test_plan_entries.size(), entry.name);
    if (entry.has_camera)
    {
        set_camera(entry.camera.position, entry.camera.yaw, entry.camera.pitch);
        app_state.camera_controller.forward = programmed_move_camera_forward(app_state.camera_controller.yaw, app_state.camera_controller.pitch);
    }
    if (entry.has_sun)
    {
        render_context.render_data.sky_settings.sun_direction = std::bit_cast<daxa_f32vec3>(programmed_move_sun_direction(entry.sun));
    }
    set_debug_draw_mode(entry.debug_draw);
    lab.presets = entry.presets;
    lab.force_full_pgi_update = entry.force_full_pgi_update;
    lab.post_stabilization_enabled = entry.post_stabilization_enabled;
    lab.post_stabilization_alpha = entry.post_stabilization_alpha;
    lab.post_stabilization_gamma = entry.post_stabilization_gamma;
    lab.post_stabilization_radius = entry.post_stabilization_radius;
    lab.post_stabilization_clamp = entry.post_stabilization_clamp;
    lab.post_stabilization_temporal_variance = entry.post_stabilization_temporal_variance;
    lab.active_preset = entry.active_preset;
    render_context.render_data.rt_lab_settings.enabled = 1;
    // Results of the previous test must not leak into this one.
    lab.has_result = false;
    lab.result = {};
    lab.move_results.clear();
    lab.measurement = {};
    lab.measure_move = entry.move;
    lab.estimate_move = entry.move;
    lab.move_warmup_frames = entry.move_warmup_frames;
    lab.move_reference_frames = entry.move_reference_frames;
    _test_plan_frame_counter = 0;
    _test_plan_test_started = false;
    _test_plan_phase = TestPlanPhase::SETTLE;
}

void Application::test_plan_finish_test(std::string const & error)
{
    TestPlanEntry const & entry = _test_plan_entries[_test_plan_index];
    nlohmann::json result = perf_test_result_json();
    result["name"] = entry.name;
    result["spec"] = entry.spec;
    f64 const wall_seconds = std::chrono::duration<f64>(std::chrono::steady_clock::now() - _test_plan_test_start_time).count();
    result["wall_seconds"] = wall_seconds;
    if (!error.empty())
    {
        result["error"] = error;
        fmt::print("[TestPlan] test \"{}\" FAILED: {}\n", entry.name, error);
    }
    _test_plan_results["tests"].push_back(std::move(result));
    std::filesystem::create_directories(_test_plan_output_dir);
    std::filesystem::path const results_path = _test_plan_output_dir / (_test_plan_name + "_results.json");
    std::ofstream(results_path) << _test_plan_results.dump(2);
    fmt::print("[TestPlan] test \"{}\" done ({:.1f} s), results: {}\n", entry.name, wall_seconds, results_path.string());
    if (entry.screenshot)
    {
        app_state.screenshot_path_override = _test_plan_output_dir / fmt::format("{}_{}_screenshot.png", _test_plan_name, entry.name);
        app_state.request_screenshot = true;
        _test_plan_phase = TestPlanPhase::SCREENSHOT;
        return;
    }
    test_plan_next();
}

void Application::test_plan_next()
{
    _test_plan_index += 1;
    if (_test_plan_index >= _test_plan_entries.size())
    {
        fmt::print("[TestPlan] \"{}\" done: {} tests\n", _test_plan_name, _test_plan_entries.size());
        _test_plan_phase = TestPlanPhase::NONE;
        app_state.keep_running = false;
        return;
    }
    test_plan_setup(_test_plan_entries[_test_plan_index]);
}

void Application::update_test_plan()
{
    auto & lab = _renderer->render_context->rt_lab;
    switch (_test_plan_phase)
    {
        case TestPlanPhase::NONE: break;
        case TestPlanPhase::LOADING:
        {
            // Same settle condition as the perf test: asset pool idle, nothing uploaded, no BLAS build queued.
            bool const settled =
                _threadpool->pending_chunk_count() == 0 &&
                !_uploaded_assets_this_frame &&
                _scene->_mesh_as_build_queue.empty();
            _test_plan_frame_counter = settled ? _test_plan_frame_counter + 1 : 0;
            if (_test_plan_frame_counter >= static_cast<i32>(PERF_TEST_SETTLE_FRAMES))
            {
                fmt::print("[TestPlan] scene loaded (frame {})\n", app_state.frame_index);
                _test_plan_index = 0;
                test_plan_setup(_test_plan_entries[0]);
            }
            break;
        }
        case TestPlanPhase::SETTLE:
        {
            TestPlanEntry const & entry = _test_plan_entries[_test_plan_index];
            _test_plan_frame_counter += 1;
            if (_test_plan_frame_counter < std::max(entry.settle_frames, 1))
            {
                break;
            }
            if (entry.type == TestPlanTestType::MEASURE)
            {
                lab.measure_frames = entry.frames;
                lab.measure_start_requested = true;
            }
            else if (entry.type == TestPlanTestType::ESTIMATE)
            {
                lab.test_frames = entry.frames;
                lab.start_requested = true;
            }
            _test_plan_frame_counter = 0;
            _test_plan_test_started = false;
            _test_plan_test_start_time = std::chrono::steady_clock::now();
            _test_plan_phase = TestPlanPhase::RUNNING;
            break;
        }
        case TestPlanPhase::RUNNING:
        {
            TestPlanEntry const & entry = _test_plan_entries[_test_plan_index];
            _test_plan_frame_counter += 1;
            if (entry.type == TestPlanTestType::PERF)
            {
                if (_test_plan_frame_counter >= entry.frames)
                {
                    test_plan_finish_test({});
                }
                break;
            }
            bool const lab_busy = lab.phase != RtLabConvergencePhase::IDLE || lab.measure_phase != RtLabMeasurePhase::IDLE;
            if (lab_busy)
            {
                _test_plan_test_started = true;
            }
            else if (_test_plan_test_started)
            {
                // Measurement: idle again after collecting its rows. Move estimate: idle once its last analysis was read.
                // Static estimate: its analysis readback arrives a few frames after it went idle -> wait for it.
                bool const waiting_for_analysis = entry.type == TestPlanTestType::ESTIMATE && !entry.move.enabled && lab.result.pixels == 0;
                if (!waiting_for_analysis || _test_plan_frame_counter > entry.frames * 2 + 64)
                {
                    test_plan_finish_test(waiting_for_analysis ? "no estimate analysis result" : std::string{});
                }
            }
            else if (_test_plan_frame_counter > 8)
            {
                test_plan_finish_test("the RT Lab test did not start");
            }
            break;
        }
        case TestPlanPhase::SCREENSHOT:
        {
            if (_perf_test_screenshot_task && _perf_test_screenshot_task->not_finished == 0)
            {
                _perf_test_screenshot_task = {};
                test_plan_next();
            }
            break;
        }
    }
}
