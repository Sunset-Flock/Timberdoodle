#include "application.hpp"

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string_view>

#if 0
#include "tex_compression/test.hpp"
int main(int argc, char const * const * argv)
{
    test_main();
}
#else
static void print_usage()
{
    fmt::print(
        "Usage: TimberDoodle [scene.gltf] [options]\n"
        "  --resolution <width> <height>        Window / render resolution (default 1024 1024).\n"
        "  --camera <x> <y> <z> <yaw> <pitch>   Teleport the camera before the scene is loaded.\n"
        "                                       Repeatable: a perf test measures every view in order.\n"
        "  --perf-test                          Load the scene, wait until all assets are streamed in, render\n"
        "                                       --frames frames, write smoothed GPU timings + a screenshot, exit.\n"
        "  --frames <n>                         Frames to render after loading (perf test, default 1000).\n"
        "  --out <dir>                          Output directory (perf test, default perf_tests).\n"
        "  --test-plan <file.json>              Run a test plan (scene, resolution and tests from the file, see\n"
        "                                       src/application_test_plan.cpp), results in <out>/<name>_results.json.\n"
        "  --name <label>                       Output file prefix (perf test, default perf).\n"
        "  --debug-draw <mode>                  Start with this DEBUG_DRAW_MODE_* index (shader_shared/shared.inl).\n"
        "  --no-specular                        Start with the RTGI specular channel disabled (A/B comparisons).\n"
        "  --upward-gloss <0..1>                Start with this RTGI 'Upward Gloss' value.\n"
        "  --rt-lab                             Start with the RT Lab denoiser instead of the RTGI denoiser.\n"
        "  --rt-lab-half-res                    Start with the RT Lab custom preset at half resolution (implies --rt-lab).\n"
        "  --rt-lab-progressive                 Start with the RT Lab progressive preset active (implies --rt-lab).\n"
        "  --rt-lab-no-spatial                  Start with the RT Lab custom preset spatial passes off (implies --rt-lab).\n"
        "  --rt-lab-atrous                      Start with the RT Lab custom preset spatial mode a-trous (implies --rt-lab).\n"
        "  --rt-lab-estimate <n>                Perf test: run an RT Lab convergence error estimate (n frames per preset)\n"
        "                                       when each view starts measuring; --frames must be > 2n. Result in the json.\n"
        "  --rt-lab-measure <n>                 Perf test: run an RT Lab convergence measurement (n frames, active preset) when\n"
        "                                       each view starts measuring; --frames must be > n + 4. Summary in the json,\n"
        "                                       result saved as rt_lab_results/<name>.rtlab.\n"
        "  --rt-lab-move <vx vy vz yaw pitch s> Programmed camera move of both RT Lab tests: camera relative velocity (m/s),\n"
        "                                       yaw / pitch speed (deg/s), duration (s). Starts from the --camera pose.\n"
        "  --rt-lab-sun-move <x y s>            Programmed sun move of both RT Lab tests: sun Angle X / Angle Y speed (deg/s),\n"
        "                                       duration (s).\n"
        "  --rt-lab-sun-start <x y>             Explicit start of the programmed sun move: sun Angle X / Angle Y (degrees).\n"
        "  --rt-lab-move-dt <s>                 Move time per step of the programmed moves (default 1/60).\n"
        "  --rt-lab-gradient-ris <0|1>          RT Lab custom preset temporal gradient sample selection: 1 RIS, 0 uniform.\n"
        "  --rt-lab-gradient <0|1>              RT Lab custom preset temporal gradient on / off.\n"
        "  --rt-lab-coarse-anti-lag <0|1>          RT Lab custom preset coarse temporal anti-lag on / off.\n"
        "  --rt-lab-move-frames <w n>           Move estimate: w custom warmup frames, n progressive frames per step\n"
        "                                       (default 256 128). Frames needed: w + steps x (n + 1) + a few.\n");
}

int main(int argc, char const * const * argv)
{
    std::optional<std::filesystem::path> scene_path = {};
    std::vector<PerfTestView> cameras = {};
    i32vec2 resolution = {1024, 1024};
    i32 debug_draw_mode = 0;
    bool no_specular = false;
    bool rt_lab = false;
    bool rt_lab_half_res = false;
    bool rt_lab_progressive = false;
    bool rt_lab_no_spatial = false;
    bool rt_lab_atrous = false;
    std::optional<i32> rt_lab_gradient_ris = {};
    std::optional<i32> rt_lab_gradient = {};
    std::optional<i32> rt_lab_coarse_anti_lag = {};
    std::optional<f32> upward_gloss = {};
    bool perf_test = false;
    std::optional<std::filesystem::path> test_plan_path = {};
    PerfTestInfo perf_test_info = {};

    for (i32 i = 1; i < argc; ++i)
    {
        std::string_view const arg = argv[i];
        auto const has_values = [&](i32 count) { return i + count < argc; };
        if (arg == "--resolution" && has_values(2))
        {
            resolution.x = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
            resolution.y = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
        }
        else if (arg == "--camera" && has_values(5))
        {
            std::array<f32, 5> values = {};
            for (f32 & v : values) { v = std::strtof(argv[++i], nullptr); }
            cameras.push_back({.position = {values[0], values[1], values[2]}, .yaw = values[3], .pitch = values[4]});
        }
        else if (arg == "--perf-test") { perf_test = true; }
        else if (arg == "--test-plan" && has_values(1)) { test_plan_path = argv[++i]; }
        else if (arg == "--no-specular") { no_specular = true; }
        else if (arg == "--rt-lab")
        {
            rt_lab = true;
        }
        else if (arg == "--rt-lab-half-res")
        {
            rt_lab = true;
            rt_lab_half_res = true;
        }
        else if (arg == "--rt-lab-measure" && has_values(1))
        {
            rt_lab = true;
            perf_test_info.rt_lab_measure_frames = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
        }
        else if (arg == "--rt-lab-estimate" && has_values(1))
        {
            rt_lab = true;
            perf_test_info.rt_lab_estimate_frames = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
        }
        else if (arg == "--rt-lab-move" && has_values(6))
        {
            auto & move = perf_test_info.rt_lab_move;
            move.enabled = true;
            move.camera.velocity.x = std::strtof(argv[++i], nullptr);
            move.camera.velocity.y = std::strtof(argv[++i], nullptr);
            move.camera.velocity.z = std::strtof(argv[++i], nullptr);
            move.camera.yaw_speed = std::strtof(argv[++i], nullptr);
            move.camera.pitch_speed = std::strtof(argv[++i], nullptr);
            move.camera.duration = std::strtof(argv[++i], nullptr);
            if (!perf_test_info.rt_lab_move_sun_set)
            {
                move.sun.duration = 0.0f;
            }
        }
        else if (arg == "--rt-lab-sun-move" && has_values(3))
        {
            auto & move = perf_test_info.rt_lab_move;
            if (!move.enabled)
            {
                move.camera.duration = 0.0f;
            }
            move.enabled = true;
            perf_test_info.rt_lab_move_sun_set = true;
            move.sun.yaw_speed = std::strtof(argv[++i], nullptr);
            move.sun.pitch_speed = std::strtof(argv[++i], nullptr);
            move.sun.duration = std::strtof(argv[++i], nullptr);
        }
        else if (arg == "--rt-lab-sun-start" && has_values(2))
        {
            auto & sun = perf_test_info.rt_lab_move.sun;
            sun.use_start_pose = true;
            sun.start.yaw = std::strtof(argv[++i], nullptr);
            sun.start.pitch = std::strtof(argv[++i], nullptr);
        }
        else if (arg == "--rt-lab-coarse-anti-lag" && has_values(1))
        {
            rt_lab = true;
            rt_lab_coarse_anti_lag = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
        }
        else if (arg == "--rt-lab-gradient" && has_values(1))
        {
            rt_lab = true;
            rt_lab_gradient = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
        }
        else if (arg == "--rt-lab-gradient-ris" && has_values(1))
        {
            rt_lab = true;
            rt_lab_gradient_ris = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
        }
        else if (arg == "--rt-lab-move-dt" && has_values(1))
        {
            perf_test_info.rt_lab_move.fixed_dt = std::max(std::strtof(argv[++i], nullptr), 1e-4f);
        }
        else if (arg == "--rt-lab-move-frames" && has_values(2))
        {
            perf_test_info.rt_lab_move_warmup_frames = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
            perf_test_info.rt_lab_move_reference_frames = static_cast<i32>(std::strtol(argv[++i], nullptr, 10));
        }
        else if (arg == "--rt-lab-progressive")
        {
            rt_lab = true;
            rt_lab_progressive = true;
        }
        else if (arg == "--rt-lab-no-spatial")
        {
            rt_lab = true;
            rt_lab_no_spatial = true;
        }
        else if (arg == "--rt-lab-atrous")
        {
            rt_lab = true;
            rt_lab_atrous = true;
        }
        else if (arg == "--upward-gloss" && has_values(1)) { upward_gloss = std::strtof(argv[++i], nullptr); }
        else if (arg == "--debug-draw" && has_values(1)) { debug_draw_mode = static_cast<i32>(std::strtol(argv[++i], nullptr, 10)); }
        else if (arg == "--frames" && has_values(1)) { perf_test_info.wait_frames = static_cast<u32>(std::strtoul(argv[++i], nullptr, 10)); }
        else if (arg == "--out" && has_values(1)) { perf_test_info.output_dir = argv[++i]; }
        else if (arg == "--name" && has_values(1)) { perf_test_info.name = argv[++i]; }
        else if (!arg.starts_with("--") && !scene_path.has_value()) { scene_path = std::filesystem::path(arg); }
        else
        {
            fmt::print("Unknown or incomplete argument \"{}\"\n", arg);
            print_usage();
            return 1;
        }
    }
    if (perf_test && !scene_path.has_value())
    {
        fmt::print("--perf-test requires a scene path\n");
        print_usage();
        return 1;
    }

    // A test plan brings its own resolution (the window is created with it).
    if (test_plan_path.has_value())
    {
        std::ifstream plan_file(test_plan_path.value());
        nlohmann::json const plan = nlohmann::json::parse(plan_file, nullptr, false);
        if (!plan_file || plan.is_discarded() || !plan.is_object())
        {
            fmt::print("--test-plan: cannot read \"{}\" as json\n", test_plan_path.value().string());
            return 1;
        }
        if (plan.contains("resolution") && plan["resolution"].is_array() && plan["resolution"].size() == 2)
        {
            resolution = {plan["resolution"][0].get<i32>(), plan["resolution"][1].get<i32>()};
        }
    }

    if (resolution.x <= 0 || resolution.y <= 0)
    {
        fmt::print("--resolution must be positive\n");
        return 1;
    }

    Application app = Application(resolution);
    app.set_debug_draw_mode(debug_draw_mode);
    if (no_specular) { app.set_rtgi_specular_enabled(false); }
    if (upward_gloss.has_value()) { app.set_upward_gloss(upward_gloss.value()); }
    if (rt_lab)
    {
        app.set_rt_lab(rt_lab_half_res, rt_lab_progressive, rt_lab_no_spatial, rt_lab_atrous);
        if (rt_lab_gradient_ris.has_value())
        {
            app.set_rt_lab_gradient_ris(rt_lab_gradient_ris.value());
        }
        if (rt_lab_gradient.has_value())
        {
            app.set_rt_lab_gradient(rt_lab_gradient.value());
        }
        if (rt_lab_coarse_anti_lag.has_value())
        {
            app.set_rt_lab_coarse_anti_lag(rt_lab_coarse_anti_lag.value());
        }
    }
    if (!cameras.empty())
    {
        app.set_camera(cameras[0].position, cameras[0].yaw, cameras[0].pitch);
    }
    if (test_plan_path.has_value())
    {
        if (!app.start_test_plan(test_plan_path.value()))
        {
            return 1;
        }
    }
    else if (perf_test)
    {
        perf_test_info.scene_path = scene_path.value();
        perf_test_info.views = cameras;
        app.start_perf_test(perf_test_info);
    }
    else if (scene_path.has_value())
    {
        app.load_scene(scene_path.value());
    }

    return app.run();
}
#endif
