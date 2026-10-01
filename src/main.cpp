#include "application.hpp"

#include <fmt/core.h>
#include <cstdlib>
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
        "  --name <label>                       Output file prefix (perf test, default perf).\n");
}

int main(int argc, char const * const * argv)
{
    std::optional<std::filesystem::path> scene_path = {};
    std::vector<PerfTestView> cameras = {};
    i32vec2 resolution = {1024, 1024};
    bool perf_test = false;
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

    if (resolution.x <= 0 || resolution.y <= 0)
    {
        fmt::print("--resolution must be positive\n");
        return 1;
    }

    Application app = Application(resolution);
    if (!cameras.empty())
    {
        app.set_camera(cameras[0].position, cameras[0].yaw, cameras[0].pitch);
    }
    if (perf_test)
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
