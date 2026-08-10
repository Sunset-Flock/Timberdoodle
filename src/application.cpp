#include "application.hpp"
#include "json_utils/camera_animation.hpp"
#include "json_utils/sky_settings.hpp"
#include <fmt/core.h>
#include <fmt/format.h>

#include <intrin.h>

// Blue noise holds raw sample values, never colour, so it loads linear.
auto load_stbn(daxa::Device & device, std::filesystem::path const & layer_zero_path) -> daxa::ImageId
{
    std::optional<daxa::ImageId> image = load_nonmanifest_texture(device, {
        .filepath = layer_zero_path,
        .layers = 64,
        .load_as_srgb = false,
    });
    if (!image.has_value())
    {
        DEBUG_MSG(fmt::format("[Renderer] ERROR failed to load Spatio Temporal Blue Noise (STBN) from path {}", layer_zero_path.string()));
        return {};
    }
    return image.value();
}

std::filesystem::path const DEFAULT_CLOUD_VOLUME_PATH = "assets\\Clouds\\default_cloud_volume.vdb";

// The three volumes a cloud is made of: the modelling fields the raymarch samples as one BC6 volume, the
// normalized SDF the custom BC1 encoder packs, and the four channel erosion noise. A .vdb naming its grids
// differently fails the cook. Authored here until the project document can carry them - the importer takes
// recipes and knows nothing about clouds.
static auto cloud_volume_recipes() -> std::vector<SlotRecipe>
{
    return {
        VdbSlotRecipe{
            .name = "cloud data",
            .grid_names = {"density", "detail_type", "density_scale"},
            .channel_mapping = {0, 1, 2},
            .target_format = daxa::Format::BC6H_UFLOAT_BLOCK,
        },
        VdbSlotRecipe{
            .name = "cloud sdf",
            .grid_names = {"sdf_normalized"},
            .channel_mapping = {0},
            .target_format = daxa::Format::BC1_RGBA_UNORM_BLOCK,
        },
        VdbSlotRecipe{
            .name = "cloud erosion noise",
            .grid_names = {"detail_noise_0", "detail_noise_1", "detail_noise_2", "detail_noise_3"},
            .channel_mapping = {0, 1, 2, 3},
            .target_format = daxa::Format::R16G16B16A16_SFLOAT,
        },
    };
}

Application::Application()
{
    _threadpool = std::make_unique<ThreadPool>(6);
    _window = std::make_unique<Window>(1024, 1024, "Timberdoodle");
    _gpu_context = std::make_unique<GPUContext>(*_window);
    // The Scene has to exist first: the Importer publishes the editor's stand-in images into it before its
    // constructor returns.
    _scene_runtime = std::make_unique<SceneRuntime>(_gpu_context->device, _gpu_context.get(), _threadpool);
    _importer = std::make_unique<Importer>(_threadpool.get(), _scene_runtime->scene());
    _ui_engine = std::make_unique<UIEngine>(*_window, _gpu_context.get());

    _renderer = std::make_unique<Renderer>(_window.get(), _gpu_context.get(), _scene_runtime->scene_ptr(), &_ui_engine->imgui_renderer, _ui_engine.get());

    std::filesystem::path const DEFAULT_SKY_SETTINGS_PATH = "settings\\sky\\default.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_sun_temple.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_san_miguel.json";
    std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_bistro.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\keypoints.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\exported_path.json";

    std::filesystem::path const STBN_BASE_PATH = "deps\\timberdoodle_assets\\STBN\\";
    _renderer->stbn2d = load_stbn(_gpu_context->device, STBN_BASE_PATH / "stbn_vec2_2Dx1D_128x128x64_0.png");
    _renderer->render_context->render_data.stbn2d = std::bit_cast<daxa_ImageViewId>(_renderer->stbn2d.default_view());
    _renderer->stbnCosDir = load_stbn(_gpu_context->device, STBN_BASE_PATH / "stbn_unitvec3_cosine_2Dx1D_128x128x64_0.png");
    _renderer->render_context->render_data.stbnCosDir = std::bit_cast<daxa_ImageViewId>(_renderer->stbnCosDir.default_view());

    _renderer->render_context->render_data.sky_settings = load_sky_settings(DEFAULT_SKY_SETTINGS_PATH);
    app_state.cinematic_camera.update_keyframes(std::move(load_camera_animation(DEFAULT_CAMERA_ANIMATION_PATH)));

    _importer->request_import(DEFAULT_CLOUD_VOLUME_PATH, cloud_volume_recipes());

    struct CompPipelinesTask : Task
    {
        Renderer * renderer = {};
        CompPipelinesTask(Renderer * renderer)
            : renderer{renderer} { chunk_count = 1; }

        virtual void callback([[maybe_unused]]u32 chunk_index, [[maybe_unused]]u32 thread_index) override
        {
            // TODO: hook up parameters.
            renderer->compile_pipelines();
        }
    };

    auto comp_pipelines_task = std::make_shared<CompPipelinesTask>(_renderer.get());

    _threadpool->async_dispatch(comp_pipelines_task);
    _threadpool->block_on(comp_pipelines_task);

    app_state.last_time_point = app_state.startup_time_point = std::chrono::steady_clock::now();
    _renderer->render_context->render_times.enable_render_times = true;
}

using FpMicroSeconds = std::chrono::duration<float, std::chrono::microseconds::period>;

void Application::load_scene(std::filesystem::path const & path)
{
    _importer->request_import(path);
}

auto Application::run() -> i32
{
    while (app_state.keep_running)
    {
        auto new_time_point = std::chrono::steady_clock::now();
        app_state.delta_time = std::chrono::duration_cast<FpMicroSeconds>(new_time_point - app_state.last_time_point).count() / 1'000'000.0f;
        app_state.last_time_point = new_time_point;
        app_state.total_elapsed_us = s_cast<u64>(std::chrono::duration_cast<FpMicroSeconds>(new_time_point - app_state.startup_time_point).count());

        {
            auto start_time_taken_cpu_windowing = std::chrono::steady_clock::now();
            _window->update(app_state.delta_time);
            app_state.keep_running &= !static_cast<bool>(glfwWindowShouldClose(_window->glfw_handle));
            i32vec2 new_window_size;
            glfwGetWindowSize(this->_window->glfw_handle, &new_window_size.x, &new_window_size.y);
            if (this->_window->size.x != new_window_size.x || _window->size.y != new_window_size.y)
            {
                this->_window->size = new_window_size;
                _renderer->window_resized();
            }
            auto end_time_taken_cpu_windowing = std::chrono::steady_clock::now();
            app_state.time_taken_cpu_windowing = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_windowing - start_time_taken_cpu_windowing).count() / 1'000'000.0f;
        }
        if (_window->size.x != 0 && _window->size.y != 0)
        {
            {
                auto start_time_taken_cpu_application = std::chrono::steady_clock::now();
                update();
                auto end_time_taken_cpu_application = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_application = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_application - start_time_taken_cpu_application).count() / 1'000'000.0f;
            }
            {
                auto start_time_taken_cpu_wait_for_gpu = std::chrono::steady_clock::now();
                _gpu_context->swapchain.wait_for_next_frame();
                auto end_time_taken_cpu_wait_for_gpu = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_wait_for_gpu = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_wait_for_gpu - start_time_taken_cpu_wait_for_gpu).count() / 1'000'000.0f;
            }
            bool execute_frame = {};
            {
                auto start_time_taken_cpu_renderer_prepare = std::chrono::steady_clock::now();
                auto const camera_info = app_state.use_preset_camera ? 
                app_state.cinematic_camera.make_camera_info(_renderer->render_context->render_data.settings) :
                app_state.camera_controller.make_camera_info(_renderer->render_context->render_data.settings);
                execute_frame = _renderer->prepare_frame(
                    app_state.frame_index,
                    camera_info,
                    app_state.observer_camera_controller.make_camera_info(_renderer->render_context->render_data.settings),
                    app_state.delta_time,
                    app_state.total_elapsed_us);
                auto end_time_taken_cpu_renderer_prepare = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_renderer_prepare = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_renderer_prepare - start_time_taken_cpu_renderer_prepare).count() / 1'000'000.0f;
            }
            if (execute_frame)
            {
                auto start_time_taken_cpu_renderer_record = std::chrono::steady_clock::now();
                _renderer->main_task_graph.execute({ 
                    .debug_ui = &_ui_engine->main_task_graph_debug_ui,
                 });
                auto end_time_taken_cpu_renderer_record = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_renderer_record = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_renderer_record - start_time_taken_cpu_renderer_record).count() / 1'000'000.0f;
            }
        }
        _gpu_context->device.collect_garbage();
        ++app_state.frame_index;
    }
    return 0;
}

void Application::update()
{
    if (!app_state.desired_scene_path.empty())
    {
        load_scene(app_state.desired_scene_path);
        app_state.desired_scene_path.clear();
    }

    _importer->tick(_scene_runtime->scene());

    // ===== Process Render Entities, Generate Mesh Instances =====

    Scene & scene = _scene_runtime->scene();
    auto const scene_instances = scene.process_entities(_renderer->render_context->render_data);
    scene.current_frame_mesh_instances = scene_instances.mesh_instances;
    scene.current_frame_cloud_volume_instances = scene_instances.cloud_volume_instances;

    // ===== Update GPU Scene Buffers =====

    scene.write_gpu_mesh_instances_buffer(scene.current_frame_mesh_instances);
    scene.write_gpu_cloud_volume_instances_buffer(scene.current_frame_cloud_volume_instances);

    usize cmd_list_count = 0ull;
    std::array<daxa::ExecutableCommandList, 16> cmd_lists = {};

    cmd_lists.at(cmd_list_count++) = _scene_runtime->update(_threadpool.get());
    cmd_lists.at(cmd_list_count++) = _scene_runtime->create_mesh_acceleration_structures();
    _gpu_context->device.submit_commands({
        .command_lists = std::span{cmd_lists.data(), cmd_list_count},
    });

    // ===== Update GPU Scene Buffers =====

    // ===== Input Handling =====

    app_state.reset_observer = false;
    if (_window->size.x == 0 || _window->size.y == 0)
    {
        return;
    }
    _ui_engine->main_update(*_renderer->render_context, _scene_runtime->scene(), app_state);
    if (_renderer->main_task_graph.get() && _ui_engine->tg_debug_ui)
    {
        _ui_engine->tg_debug_ui = _ui_engine->main_task_graph_debug_ui.update(_renderer->main_task_graph);
    }
    if (app_state.use_preset_camera)
    {
        app_state.cinematic_camera.process_input(*_window, app_state.delta_time);
    }
    if (app_state.control_observer) {
        app_state.observer_camera_controller.process_input(*_window, app_state.delta_time);
    }
    else {
        app_state.camera_controller.process_input(*_window, app_state.delta_time);
    }

    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        if (_window->key_just_pressed(GLFW_KEY_H))
        {
            _renderer->render_context->render_data.settings.draw_from_observer = !_renderer->render_context->render_data.settings.draw_from_observer;
        }
        app_state.cinematic_camera.override_keyframe = 
            _window->key_just_pressed(GLFW_KEY_I) ?
            !app_state.cinematic_camera.override_keyframe :
            app_state.cinematic_camera.override_keyframe;
        if (_window->key_just_pressed(GLFW_KEY_J)) { app_state.control_observer = !app_state.control_observer; }
        if (_window->key_just_pressed(GLFW_KEY_K)) { app_state.reset_observer = true; }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->button_just_pressed(GLFW_MOUSE_BUTTON_1))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position = {
                _window->get_cursor_x(),
                _window->get_cursor_y(),
            };
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_LEFT))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.x -= 1;
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_RIGHT))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.x += 1;
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_UP))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.y -= 1;
        }
        if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_DOWN))
        {
            _renderer->gpu_context->shader_debug_context.detector_window_position.y += 1;
        }
    }

    if (app_state.reset_observer)
    {
        app_state.control_observer = false;
        _renderer->render_context->render_data.settings.draw_from_observer = static_cast<u32>(false);
        app_state.observer_camera_controller = app_state.camera_controller;
    }

    // ===== Input Handling =====
}


Application::~Application()
{
    // Joins the in-flight parse/cook/cache-write tasks. The Importer is still alive and still holds them, so
    // they have somewhere to write; nothing polls them again, so what they produce is simply dropped.
    _threadpool.reset();
    // Thread pool is gone here: update won't spawn new texture streams, just flushes GPU updates.
    auto manifest_update_commands = _scene_runtime->update(nullptr);
    auto cmd_lists = std::array{std::move(manifest_update_commands)};
    _gpu_context->device.submit_commands({.command_lists = cmd_lists});
    _gpu_context->device.wait_idle();
}