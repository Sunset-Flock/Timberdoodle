#include "application.hpp"
#include "rendering/rtgi/rt_lab/rt_lab.hpp"
#include "rendering/rtgi/rt_lab/rt_lab2.hpp"
#include "rendering/rtgi/rt_lab/rt_lab3.hpp"
#include "rendering/rtgi/rt_lab/rt_lab4.hpp"
#include "rendering/rtgi/rt_lab/rt_lab_integrator.hpp"

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
//       "test": "lab" | "measure" | "estimate" | "perf",    default measure. lab / measure / estimate = RT Lab test
//                                                           (RtLabTestSetup), they only differ in the default metrics:
//                                                           lab / measure = convergence + instability + lag,
//                                                           estimate = reference only
//       "metrics": ["convergence", "instability", "lag", "reference"],   RT Lab test metrics (overrides the default)
//       "frames": 200,                                      RT Lab: tested frames; perf: frames before the timings
//       "reference_frames": 200,                            static reference: progressive frames (default = frames)
//       "settle_frames": 16,                                frames rendered after the setup, before the test starts
//       "camera": "px py pz fx fy fz" | [x, y, z, yaw, pitch],   View Camera "Copy to clipboard" format or yaw / pitch
//       "sun": "angle_x angle_y" | [angle_x, angle_y],      sun panel angles (degrees)
//       "debug_draw": 0,                                    DEBUG_DRAW_MODE_* (int)
//       "preset": "custom" | "progressive" | "neural",      active (tested) RT Lab preset, default custom
//       "force_full_pgi_update": false,                     RT Lab "Force Full PGI Update" for this test
//       "lag_tolerance": 0.02,                              temporal lag metric tolerance (relative), default 0.02
//       "custom": { "coarse_anti_lag_enabled": 1, ... },       RT Lab preset overrides, keys = saved result settings keys,
//       "progressive": { ... }, "neural_preset": { ... },   applied to the app's presets as they were at plan start (the neural preset also takes "custom" first)
//       "lab2": { "enabled": true, "ray_budget": 0.25, "cell_size_px": 16, ... },   RT Lab 2 (hash grid denoiser) instead of the original denoiser:
//                                                           keys = RtLab2Settings field names (rt_lab2_settings.inl), default off. The progressive reference
//                                                           phases still run the original denoiser; every test starts from the RT Lab 2 defaults
//       "integrator": { "enabled": true, "window_radius": 2, ... },   THE temporal resolve (RtLabIntegratorSettings field names, the hash grid's integrator), used by every
//                                                           pipeline; part of the preset config, default ON (progressive preset: off). The integrator_* / window_radius ...
//                                                           keys are no longer RT Lab 2 keys.
//       The "lab2" / "lab3" / "lab4" / "integrator" objects are applied to the ACTIVE preset's config (RtLabState::preset_configs), defaults otherwise.
//       pipeline: part of the preset settings, e.g. "custom": { "restir_enabled": 1, "denoiser": 0 }   denoiser 0 classic, 1 hash grid, 2 none (ReSTIR only).
//                                                           The "enabled" flags of lab2 / lab3 / lab4 below still work (they select version 2 / 3 / 4 in the active preset).
//       "lab4": { "enabled": true, "cell_source": 0, "cell_weight_mode": 1 },   RT Lab 4 = RT Lab 3 (ReSTIR GI, "lab3" settings) in front of RT Lab 2 (cells + temporal resolve,
//                                                           "lab2" settings, no temporal gradient); default off, takes priority over "lab3" / "lab2".
//                                                           cell_weight_mode: 0 one per pixel with a reservoir, 1 fresh rays, 2 reservoir confidence M
//       "lab3": { "enabled": true, "ray_budget": 1.0, "spatial_passes": 1, ... },   RT Lab 3 (ReSTIR GI): keys = "enabled" (legacy), "half_res" (sets the preset),
//                                                           "ray_budget", "redistribution_enabled", "max_rays_per_pixel", "target_sample_count", "min_base_coverage"
//                                                           and the RtLab3Settings field names (rt_lab3_settings.inl); default off, takes priority over "lab2"
//       "move": {                                           programmed move of the RT Lab test
//         "dt": 0.0166667,                                  time per step
//         "camera": { "velocity": [vx, vy, vz], "yaw_speed": 0, "pitch_speed": 0, "duration": 1, "start": "px py pz fx fy fz" },
//         "sun": { "yaw_speed": -25, "pitch_speed": -35, "duration": 2, "start": "250 90" }
//       },                                                  a missing camera / sun move = none (duration 0)
//       "move_frames": [256, 128],                          move reference: tested warmup, progressive frames per step
//       "light_shock": [start, frames],                     temporal shock: point light culling flipped for run frames
//                                                           [start, start + frames), lag metric from its end
//       "neural": {                                         RT Lab neural affinity (rendering/rtgi/rt_lab/neural). enabled / coopvec / half /
//         "enabled": true, "coopvec": true, "half": false,   target_samples are legacy keys: they set the ACTIVE preset's neural_* settings
//         "target_samples": 1000000,                        (better: preset overrides "neural_enabled", "neural_target_samples", ... or preset "neural")
//         "weights": "src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural.bin",
//         "dump": { "first": 0, "frames": 96, "crop": [x, y, 192], "path": "<out>/<name>_<test>.rtlndump" }   training data
//                    ("frame_list": [0, 3, 7, 15, 31, 63] instead of first / frames = the RUN frames to dump)
//       },                                                  (dump: needs frames > first + frames + 8; train/dataset.py reads it)
//       "screenshot": false                                 <out>/<name>_<test>_screenshot.png after the test
//     }
//   ]
// }
// Per test result: the perf test blocks (camera, rt_lab_measure / rt_lab_convergence (+ move_steps), groups = GPU
// timings) + "name", "spec" (the merged entry), "wall_seconds", "error" when the test failed.

// Test type + the RT Lab test's default metrics.
static auto test_plan_type(std::string const & type, TestPlanTestType & out, u32 & metrics) -> bool
{
    out = TestPlanTestType::RT_LAB;
    metrics = RT_LAB_TEST_METRIC_DEFAULT;
    if (type == "perf")
    {
        out = TestPlanTestType::PERF;
    }
    else if (type == "estimate")
    {
        metrics = RT_LAB_TEST_METRIC_REFERENCE;
    }
    else if (type != "lab" && type != "measure")
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
        if (!test_plan_type(spec.value("test", std::string("measure")), entry.type, entry.test.metrics))
        {
            return fail("\"test\" must be lab, measure, estimate or perf");
        }
        if (spec.contains("metrics"))
        {
            if (!spec["metrics"].is_array())
            {
                return fail("\"metrics\" must be an array of names");
            }
            std::string names = {};
            for (nlohmann::json const & name : spec["metrics"])
            {
                names += name.get<std::string>() + ",";
            }
            std::string unknown = {};
            if (!rt_lab_test_metrics_from_names(names, entry.test.metrics, unknown))
            {
                return fail(fmt::format("unknown metric \"{}\"", unknown));
            }
        }
        entry.frames = spec.value("frames", entry.frames);
        entry.test.frames = entry.frames;
        entry.test.reference_frames = spec.value("reference_frames", entry.frames);
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
        if (preset != "custom" && preset != "progressive" && preset != "neural")
        {
            return fail("\"preset\" must be custom, progressive or neural");
        }
        entry.active_preset = preset == "custom" ? RT_LAB_PRESET_CUSTOM : (preset == "neural" ? RT_LAB_PRESET_NEURAL : RT_LAB_PRESET_PROGRESSIVE);
        for (i32 p = 0; p < RT_LAB_PRESET_COUNT; ++p)
        {
            char const * const key = p == RT_LAB_PRESET_CUSTOM ? "custom" : (p == RT_LAB_PRESET_NEURAL ? "neural_preset" : "progressive");
            std::vector<std::string> unknown = {};
            entry.presets[p] = lab.presets[p];
            if (p == RT_LAB_PRESET_NEURAL)
            {
                // The Neural preset starts from the plan's custom overrides, then its own ("neural" is the affinity config).
                entry.presets[p] = rt_lab_settings_apply_json(entry.presets[p], spec.value("custom", nlohmann::json::object()), unknown);
            }
            entry.presets[p] = rt_lab_settings_apply_json(entry.presets[p], spec.value(key, nlohmann::json::object()), unknown);
            if (!unknown.empty())
            {
                return fail(fmt::format("unknown \"{}\" setting \"{}\"", key, unknown.front()));
            }
        }
        if (spec.contains("lab2"))
        {
            RtLab2State scratch = {};
            std::vector<std::string> unknown = {};
            rt_lab2_apply_json(scratch, spec["lab2"], unknown);
            if (!unknown.empty())
            {
                return fail(fmt::format("unknown \"lab2\" setting \"{}\"", unknown.front()));
            }
        }
        if (spec.contains("integrator"))
        {
            RtLabIntegratorState scratch = {};
            std::vector<std::string> unknown = {};
            rt_lab_integrator_apply_json(scratch, spec["integrator"], unknown);
            if (!unknown.empty())
            {
                return fail(fmt::format("unknown \"integrator\" setting \"{}\"", unknown.front()));
            }
        }
        if (spec.contains("lab4"))
        {
            RtLab4State scratch = {};
            std::vector<std::string> unknown = {};
            rt_lab4_apply_json(scratch, spec["lab4"], unknown);
            if (!unknown.empty())
            {
                return fail(fmt::format("unknown \"lab4\" setting \"{}\"", unknown.front()));
            }
        }
        if (spec.contains("lab3"))
        {
            RtLab3State scratch = {};
            std::vector<std::string> unknown = {};
            rt_lab3_apply_json(scratch, spec["lab3"], unknown);
            if (!unknown.empty())
            {
                return fail(fmt::format("unknown \"lab3\" setting \"{}\"", unknown.front()));
            }
        }
        if (spec.contains("move"))
        {
            nlohmann::json const & move = spec["move"];
            RtLabTestMove & test_move = entry.test.move;
            test_move.enabled = true;
            test_move.fixed_dt = std::max(move.value("dt", 1.0f / 60.0f), 1e-4f);
            test_move.camera = move.contains("camera") ? programmed_move_from_json(move["camera"], ProgrammedMove{.duration = 0.0f}) : ProgrammedMove{.duration = 0.0f};
            test_move.sun = move.contains("sun") ? programmed_move_from_json(move["sun"], ProgrammedMove{.translation = false, .duration = 0.0f})
                                                 : ProgrammedMove{.translation = false, .duration = 0.0f};
            test_move.camera.translation = true;
            test_move.sun.translation = false;
        }
        if (spec.contains("move_frames") && spec["move_frames"].is_array() && spec["move_frames"].size() == 2)
        {
            entry.test.move_warmup_frames = spec["move_frames"][0].get<i32>();
            entry.test.move_reference_frames = spec["move_frames"][1].get<i32>();
        }
        if (spec.contains("light_shock") && spec["light_shock"].is_array() && spec["light_shock"].size() == 2)
        {
            entry.test.shock_start = std::max(spec["light_shock"][0].get<i32>(), 0);
            entry.test.shock_frames = std::max(spec["light_shock"][1].get<i32>(), 0);
        }
        entry.screenshot = spec.value("screenshot", false);
        entry.force_full_pgi_update = spec.value("force_full_pgi_update", false);
        entry.lag_tolerance = spec.value("lag_tolerance", 0.02f);
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
    lab.lag_tolerance = entry.lag_tolerance;
    lab.active_preset = entry.active_preset;
    {
        // The stage settings (hash grid, ReSTIR, glue, temporal resolve) are part of the active preset's config: tests are independent, every test starts from
        // the preset's defaults plus its own "lab2" / "lab3" / "lab4" / "integrator" overrides.
        auto const section = [&](char const * key) -> nlohmann::json
        {
            return entry.spec.value(key, nlohmann::json::object());
        };
        RtLabPresetConfig config = rt_lab_default_preset_config(static_cast<RtLabPreset>(std::clamp(lab.active_preset, 0, RT_LAB_PRESET_COUNT - 1)));
        std::vector<std::string> unknown = {};
        RtLab2State lab2 = {};
        rt_lab2_apply_json(lab2, section("lab2"), unknown);
        config.lab2 = lab2;
        RtLab3State lab3 = {};
        rt_lab3_apply_json(lab3, section("lab3"), unknown);
        config.lab3 = lab3;
        RtLab4State lab4 = {};
        rt_lab4_apply_json(lab4, section("lab4"), unknown);
        config.lab4 = lab4;
        RtLabIntegratorState integrator = {};
        integrator.enabled = config.integrator.enabled;
        rt_lab_integrator_apply_json(integrator, section("integrator"), unknown);
        config.integrator = integrator;
        lab.preset_configs[lab.active_preset] = config;
        if (section("lab3").contains("half_res"))
        {
            lab.presets[lab.active_preset].half_res = lab3.half_res ? 1 : 0; // the resolution is a preset setting now
        }
        // Legacy "enabled" flags of lab2 / lab3 / lab4 (4 wins over 3 over 2): select that version in the active preset (the preset keys "restir_enabled" /
        // "denoiser" of "custom" etc. are applied with the presets and win when the plan sets them).
        i32 const legacy = lab4.enabled ? 4 : (lab3.enabled ? 3 : (lab2.enabled ? 2 : 0));
        if (legacy != 0 && !section("custom").contains("denoiser") && !section("custom").contains("restir_enabled"))
        {
            rt_lab_pipeline_set_version(lab.presets[lab.active_preset], legacy);
        }
    }
    {
        // Neural affinity + training data dump (rendering/rtgi/rt_lab/neural).
        nlohmann::json const neural = entry.spec.value("neural", nlohmann::json::object());
        // Plans from before the neural settings lived in the presets: these keys set the ACTIVE preset's neural_* settings.
        RtLabSettings & active = lab.presets[std::clamp(lab.active_preset, 0, RT_LAB_PRESET_COUNT - 1)];
        active.neural_enabled = neural.value("enabled", active.neural_enabled != 0) ? 1 : 0;
        active.neural_use_coopvec = neural.value("coopvec", active.neural_use_coopvec != 0) ? 1 : 0;
        active.neural_use_half = neural.value("half", active.neural_use_half != 0) ? 1 : 0;
        active.neural_target_samples = neural.value("target_samples", active.neural_target_samples);
        if (neural.contains("weights"))
        {
            lab.neural.weights_path = neural["weights"].get<std::string>();
        }
        lab.neural.dump.enabled = false;
        if (neural.contains("dump"))
        {
            nlohmann::json const & dump = neural["dump"];
            std::filesystem::path const path = dump.contains("path") ? std::filesystem::path(dump["path"].get<std::string>())
                                                                      : _test_plan_output_dir / fmt::format("{}_{}.rtlndump", _test_plan_name, entry.name);
            nlohmann::json const crop = dump.value("crop", nlohmann::json::array({0, 0, 192}));
            rt_lab_neural_dump_begin(lab.neural, path, dump.value("first", 0), dump.value("frames", 96), crop.size() > 0 ? crop[0].get<i32>() : 0,
                crop.size() > 1 ? crop[1].get<i32>() : 0, crop.size() > 2 ? crop[2].get<i32>() : 192, dump.value("frame_list", std::vector<i32>{}));
        }
    }
    render_context.render_data.rt_lab_settings.enabled = 1;
    // Results of the previous test must not leak into this one.
    lab.measurement = {};
    lab.live_analysis = false;
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
            if (entry.type == TestPlanTestType::RT_LAB)
            {
                lab.test = entry.test;
                lab.test_start_requested = true;
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
                // "sun_flip": ["angle_x angle_y", "angle_x angle_y"]: the sun alternates between both directions EVERY frame (every frame is a jump frame).
                if (entry.spec.contains("sun_flip") && entry.spec["sun_flip"].is_array() && entry.spec["sun_flip"].size() == 2)
                {
                    ProgrammedMovePose flip_pose = {};
                    if (test_plan_pose(entry.spec["sun_flip"][_test_plan_frame_counter & 1], false, flip_pose))
                    {
                        _renderer->render_context->render_data.sky_settings.sun_direction = std::bit_cast<daxa_f32vec3>(programmed_move_sun_direction(flip_pose));
                    }
                }
                // "sun_jumps": [[frame, "angle_x angle_y"], ...]: the sun teleports at that frame of the test (instant lighting change).
                if (entry.spec.contains("sun_jumps") && entry.spec["sun_jumps"].is_array())
                {
                    for (nlohmann::json const & jump : entry.spec["sun_jumps"])
                    {
                        ProgrammedMovePose pose = {};
                        if (jump.is_array() && jump.size() == 2 && jump[0].get<i32>() == _test_plan_frame_counter && test_plan_pose(jump[1], false, pose))
                        {
                            _renderer->render_context->render_data.sky_settings.sun_direction = std::bit_cast<daxa_f32vec3>(programmed_move_sun_direction(pose));
                        }
                    }
                }
                if (_test_plan_frame_counter >= entry.frames)
                {
                    test_plan_finish_test({});
                }
                break;
            }
            if (rt_lab_test_running(lab))
            {
                _test_plan_test_started = true;
            }
            else if (_test_plan_test_started)
            {
                // Idle again once the test collected its rows and its reference analysis.
                bool const missing_reference = (lab.measurement.metrics & RT_LAB_TEST_METRIC_REFERENCE) != 0 && lab.measurement.reference.pixels == 0;
                test_plan_finish_test(missing_reference ? "no reference analysis result" : std::string{});
            }
            else if (_test_plan_frame_counter > 8)
            {
                test_plan_finish_test("the RT Lab test did not start");
            }
            break;
        }
        case TestPlanPhase::SCREENSHOT:
        {
            // "sun_jump_late": "angle_x angle_y": the sun teleports one frame after the screenshot request = the screenshotted frame is the jump frame.
            TestPlanEntry const & shot_entry = _test_plan_entries[_test_plan_index];
            _test_plan_frame_counter += 1;
            ProgrammedMovePose late_pose = {};
            if (shot_entry.spec.contains("sun_jump_late") && _test_plan_frame_counter == shot_entry.frames + 1 &&
                test_plan_pose(shot_entry.spec["sun_jump_late"], false, late_pose))
            {
                _renderer->render_context->render_data.sky_settings.sun_direction = std::bit_cast<daxa_f32vec3>(programmed_move_sun_direction(late_pose));
            }
            if (_perf_test_screenshot_task && _perf_test_screenshot_task->not_finished == 0)
            {
                _perf_test_screenshot_task = {};
                test_plan_next();
            }
            break;
        }
    }
}
