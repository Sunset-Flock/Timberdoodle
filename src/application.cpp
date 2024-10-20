#include "application.hpp"
#include "json_handler.hpp"
#include <fmt/core.h>
#include <fmt/format.h>

#include <intrin.h>

Application::Application()
{
    _threadpool = std::make_unique<ThreadPool>(7);
    _window = std::make_unique<Window>(1024, 1024, "Sandbox");
    _gpu_context = std::make_unique<GPUContext>(*_window);
    _scene = std::make_unique<Scene>(_gpu_context->device, _gpu_context.get());
    _asset_manager = std::make_unique<AssetProcessor>(_gpu_context->device);
    _ui_engine = std::make_unique<UIEngine>(*_window, *_asset_manager, _gpu_context.get());

    _renderer = std::make_unique<Renderer>(_window.get(), _gpu_context.get(), _scene.get(), _asset_manager.get(), &_ui_engine->imgui_renderer, _ui_engine.get());

    std::filesystem::path const DEFAULT_SKY_SETTINGS_PATH = "settings\\sky\\default.json";
    std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path.json";

    _renderer->render_context->render_data.sky_settings = load_sky_settings(DEFAULT_SKY_SETTINGS_PATH);
    app_state.cinematic_camera.update_keyframes(std::move(load_camera_animation(DEFAULT_CAMERA_ANIMATION_PATH)));

    struct CompPipelinesTask : Task
    {
        Renderer * renderer = {};
        CompPipelinesTask(Renderer * renderer)
            : renderer{renderer} { chunk_count = 1; }

        virtual void callback(u32 chunk_index, u32 thread_index) override
        {
            // TODO: hook up parameters.
            renderer->compile_pipelines();
        }
    };

    auto comp_pipelines_task = std::make_shared<CompPipelinesTask>(_renderer.get());

    _threadpool->async_dispatch(comp_pipelines_task);
    _threadpool->block_on(comp_pipelines_task);

    // TODO(ui): DO NOT ALWAYS JUST LOAD THIS UNCONDITIONALLY!
    // TODO(ui): ADD UI FOR LOADING IN THE EDITOR!
    std::filesystem::path const DEFAULT_HARDCODED_PATH = ".\\assets";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro\\bistro.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro_compressed\\bistro_c.gltf";
    std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro_fix_ball_compressed\\bistro_fix_ball_c.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "medium\\medium.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "hermitcraft\\large.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "bunnies\\bunnies2.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "battle_scene_compressed\\battle_scene_c.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "cube/cube.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "TestWorld\\TestWorld.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "repro\\minimal.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "flying_world\\flying_world.gltf";

    auto const result = _scene->load_manifest_from_gltf({
        .root_path = DEFAULT_HARDCODED_PATH,
        .asset_name = DEFAULT_HARDCODED_FILE,
        .thread_pool = _threadpool,
        .asset_processor = _asset_manager,
    });

    if (Scene::LoadManifestErrorCode const * err = std::get_if<Scene::LoadManifestErrorCode>(&result))
    {
        DEBUG_MSG(fmt::format("[WARN][Application::Application()] Loading \"{}\" Error: {}",
            (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string(), Scene::to_string(*err)));
    }
    // TODO(msakmary) clean this up!
    else
    {
        auto const r_id = std::get<RenderEntityId>(result);
        RenderEntity & r_ent = *_scene->_render_entities.slot(r_id);

        auto child = r_ent.first_child;
        RenderEntity * bistro_exterior = {};
        while (child.has_value())
        {
            auto child_node = *_scene->_render_entities.slot(child.value());
            if (child_node.name == "BistroExterior")
            {
                bistro_exterior = &child_node;
                break;
            }
            child = child_node.next_sibling;
        }
        if (bistro_exterior)
        {
            child = bistro_exterior->first_child;
            while (child.has_value())
            {
                auto child_node = *_scene->_render_entities.slot(child.value());
                if (child_node.name == "DYNAMIC_sphere")
                {
                    app_state.dynamic_ball = child.value();
                    break;
                }
                child = child_node.next_sibling;
            }
        }

        DEBUG_MSG(fmt::format("[INFO][Application::Application()] Loading \"{}\" Success",
            (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string()));
    }

    app_state.last_time_point = std::chrono::steady_clock::now();
}
using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;

auto Application::run() -> i32
{
    while (app_state.keep_running)
    {
        auto new_time_point = std::chrono::steady_clock::now();
        app_state.delta_time = std::chrono::duration_cast<FpMilliseconds>(new_time_point - app_state.last_time_point).count() * 0.001f;
        app_state.last_time_point = new_time_point;
        _window->update(app_state.delta_time);
        app_state.keep_running &= !static_cast<bool>(glfwWindowShouldClose(_window->glfw_handle));
        i32vec2 new_window_size;
        glfwGetWindowSize(this->_window->glfw_handle, &new_window_size.x, &new_window_size.y);
        if (this->_window->size.x != new_window_size.x || _window->size.y != new_window_size.y)
        {
            this->_window->size = new_window_size;
            _renderer->window_resized();
        }
        if (_window->size.x != 0 && _window->size.y != 0)
        {
            update();
            auto const camera_info = app_state.use_preset_camera ? 
                app_state.cinematic_camera.make_camera_info(_renderer->render_context->render_data.settings) :
                app_state.camera_controller.make_camera_info(_renderer->render_context->render_data.settings);

            _renderer->render_frame(
                camera_info,
                app_state.observer_camera_controller.make_camera_info(_renderer->render_context->render_data.settings),
                app_state.delta_time);
        }
        _gpu_context->device.collect_garbage();
    }
    return 0;
}

void Application::update()
{
    auto * dynamic_ball_ent = _scene->_render_entities.slot(app_state.dynamic_ball);
    if (dynamic_ball_ent)
    {
        auto prev_transform = glm::mat4(
            glm::vec4(dynamic_ball_ent->transform[0], 0.0f),
            glm::vec4(dynamic_ball_ent->transform[1], 0.0f),
            glm::vec4(dynamic_ball_ent->transform[2], 0.0f),
            glm::vec4(dynamic_ball_ent->transform[3], 1.0f));

        static f32 total_time = 0.0f;
        total_time += app_state.delta_time;
        auto new_position = f32vec4{
            std::sin(total_time) * 100.0f,
            std::cos(total_time) * 100.0f,
            prev_transform[3].z,
            1.0f};
        auto curr_transform = prev_transform;
        curr_transform[3] = new_position;

        dynamic_ball_ent->transform = curr_transform;
        _scene->_modified_render_entities.push_back({app_state.dynamic_ball, prev_transform, curr_transform});
    }

    auto asset_data_upload_info = _asset_manager->record_gpu_load_processing_commands();
    auto manifest_update_commands = _scene->record_gpu_manifest_update({
        .uploaded_meshes = asset_data_upload_info.uploaded_meshes,
        .uploaded_textures = asset_data_upload_info.uploaded_textures,
    });
    auto tmp_cpu_mesh_instances = _scene->process_entities();
    _scene->write_gpu_mesh_instances_buffer(std::move(tmp_cpu_mesh_instances));

    bool const merged_blas = _renderer->render_context->render_data.settings.enable_merged_scene_blas;
    auto rt_merged_update_commands = _scene->create_merged_as_and_record_build_commands(merged_blas);
    auto rt_update_commands = _scene->create_as_and_record_build_commands(!merged_blas);
    auto cmd_lists = std::array{
        std::move(asset_data_upload_info.upload_commands),
        std::move(manifest_update_commands),
        std::move(rt_merged_update_commands),
        std::move(rt_update_commands),
    };
    _gpu_context->device.submit_commands({.command_lists = cmd_lists});

    app_state.reset_observer = false;
    if (_window->size.x == 0 || _window->size.y == 0)
    {
        return;
    }
    _ui_engine->main_update(*_gpu_context, *_renderer->render_context, *_scene, app_state);
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
    if (_window->key_just_pressed(GLFW_KEY_O)) {
        DEBUG_MSG(fmt::format("switched observer_show_pass from {} to {}", _renderer->render_context->render_data.settings.observer_show_pass,
            ((_renderer->render_context->render_data.settings.observer_show_pass + 1) % 3)));
        _renderer->render_context->render_data.settings.observer_show_pass = (_renderer->render_context->render_data.settings.observer_show_pass + 1) % 3;
    }
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

    if (app_state.reset_observer)
    {
        app_state.control_observer = false;
        _renderer->render_context->render_data.settings.draw_from_observer = static_cast<u32>(false);
        app_state.observer_camera_controller = app_state.camera_controller;
    }
}


Application::~Application()
{
    _threadpool.reset();
    auto asset_data_upload_info = _asset_manager->record_gpu_load_processing_commands();
    auto manifest_update_commands = _scene->record_gpu_manifest_update({
        .uploaded_meshes = asset_data_upload_info.uploaded_meshes,
        .uploaded_textures = asset_data_upload_info.uploaded_textures,
    });
    auto cmd_lists = std::array{std::move(asset_data_upload_info.upload_commands), std::move(manifest_update_commands)};
    _gpu_context->device.submit_commands({.command_lists = cmd_lists});
    _gpu_context->device.wait_idle();
}