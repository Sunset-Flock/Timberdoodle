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
    std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_sun_temple.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_san_miguel.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\cam_path_bistro.json";
    // std::filesystem::path const DEFAULT_CAMERA_ANIMATION_PATH = "settings\\camera\\keypoints.json";

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

    app_state.last_time_point = app_state.startup_time_point = std::chrono::steady_clock::now();
    _renderer->render_context->render_times.enable_render_times = true;
}

using FpMicroSeconds = std::chrono::duration<float, std::chrono::microseconds::period>;

void Application::load_scene(std::filesystem::path const & path)
{
    if (!path.has_filename() || !path.has_parent_path())
    {
        return;
    }

    auto const result = _scene->load_manifest_from_gltf({
        .root_path = path.parent_path(),
        .asset_name = path.filename(),
        .thread_pool = _threadpool,
        .asset_processor = _asset_manager,
    });

    if (Scene::LoadManifestErrorCode const * err = std::get_if<Scene::LoadManifestErrorCode>(&result))
    {
        DEBUG_MSG(fmt::format("[WARN][Application::Application()] Loading \"{}\" Error: {}",
            path.string(), Scene::to_string(*err)));
    }
    // TODO(msakmary) HACKY - fix this
    // =========================================================================
    else
    {
        auto const r_id = std::get<RenderEntityId>(result);
        app_state.root_id = r_id;
        RenderEntity & r_ent = *_scene->_render_entities.slot(r_id);

        for (u32 entity_i = 0; entity_i < _scene->_render_entities.capacity(); ++entity_i)
        {
            RenderEntity const * r_ent = _scene->_render_entities.slot_by_index(entity_i);
            if (r_ent->name == "DYNAMIC_sphere")
            {
                app_state.dynamic_ball = _scene->_render_entities.id_from_index(entity_i);
            }
        }

        DEBUG_MSG(fmt::format("[INFO][Application::Application()] Loading \"{}\" Success", path.string()));
    }
    // =========================================================================
}

auto Application::run() -> i32
{
    while (app_state.keep_running)
    {
        auto new_time_point = std::chrono::steady_clock::now();
        app_state.delta_time = std::chrono::duration_cast<FpMicroSeconds>(new_time_point - app_state.last_time_point).count() / 1'000'000.0f;
        app_state.last_time_point = new_time_point;
        app_state.total_elapsed_us = std::chrono::duration_cast<FpMicroSeconds>(new_time_point - app_state.startup_time_point).count();

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
                _renderer->main_task_graph.execute({});
                auto end_time_taken_cpu_renderer_record = std::chrono::steady_clock::now();
                app_state.time_taken_cpu_renderer_record = std::chrono::duration_cast<FpMicroSeconds>(end_time_taken_cpu_renderer_record - start_time_taken_cpu_renderer_record).count() / 1'000'000.0f;
            }
        }
        _gpu_context->device.collect_garbage();
    }
    return 0;
}

void Application::update()
{
    if (!app_state.desired_scene_path.empty())
    {
        fmt::print("Requested load: {}\n", app_state.desired_scene_path);
        load_scene(app_state.desired_scene_path);
        app_state.desired_scene_path.clear();
    }

    // TODO(msakmary) HACKY - fix this
    // ===== Saky's Ball =====
    {
        auto mat_4x3_to_4x4 = [](glm::mat4x3 const & transform) -> glm::mat4x4
        {
            return glm::mat4x4{
                glm::vec4(transform[0], 0.0f),
                glm::vec4(transform[1], 0.0f),
                glm::vec4(transform[2], 0.0f),
                glm::vec4(transform[3], 1.0f)};
        };
        static f32 total_time = 0.0f;
        total_time += app_state.delta_time;

        // {
        //     RenderEntity * r_ent = _scene->_render_entities.slot(app_state.root_id);
        //     auto transform = mat_4x3_to_4x4(r_ent->transform);
        //     transform = glm::rotate(transform, glm::radians(90.0f), f32vec3(1.0f, 0.0f, 0.0f));

        //     r_ent->transform = transform;
        //     _scene->_dirty_render_entities.push_back(app_state.root_id);
        // }

        auto * dynamic_ball_ent = _scene->_render_entities.slot(app_state.dynamic_ball);
        // if (dynamic_ball_ent)
        if (false)
        {
            auto prev_transform = mat_4x3_to_4x4(dynamic_ball_ent->transform);

            auto new_position = f32vec4{
                std::sin(total_time) * 100.0f,
                std::cos(total_time) * 100.0f,
                prev_transform[3].z,
                1.0f};
            auto curr_transform = prev_transform;
            curr_transform[3] = new_position;

            dynamic_ball_ent->transform = curr_transform;
            _scene->_dirty_render_entities.push_back(app_state.dynamic_ball);
        }

        if(app_state.decompose_bistro) 
        {
            for (u32 entity_i = 0; entity_i < _scene->_render_entities.capacity(); ++entity_i)
            {
                RenderEntity * r_ent = _scene->_render_entities.slot_by_index(entity_i);
                if(r_ent->mesh_group_manifest_index.has_value())// && strstr(r_ent->name.c_str(), "StreetLight"))
                {
                    auto transform = mat_4x3_to_4x4(r_ent->transform);

                    transform = transform * glm::inverse(mat_4x3_to_4x4(_scene->_render_entities.slot(r_ent->parent.value())->combined_transform));
                    transform = glm::rotate(transform, glm::radians(sin(total_time * 0.00001f) * 50.0f), glm::normalize(glm::vec3(0.0, 1.0, 0.0)));
                    transform = transform * mat_4x3_to_4x4(_scene->_render_entities.slot(r_ent->parent.value())->combined_transform);

                    r_ent->transform = transform;
                    _scene->_dirty_render_entities.push_back(_scene->_render_entities.id_from_index(entity_i));
                }
            }
        }
    }
    // ===== Saky's Ball =====

    // ===== Process Render Entities, Generate Mesh Instances =====

    _scene->current_frame_mesh_instances = _scene->process_entities(_renderer->render_context->render_data);

    // ===== Process Render Entities, Generate Mesh Instances =====

    // ===== Update GPU Scene Buffers =====

    _scene->write_gpu_mesh_instances_buffer(_scene->current_frame_mesh_instances);

    usize cmd_list_count = 0ull;
    std::array<daxa::ExecutableCommandList, 16> cmd_lists = {};

    auto asset_data_upload_info = _asset_manager->record_gpu_load_processing_commands();
    cmd_lists.at(cmd_list_count++) = std::move(asset_data_upload_info.upload_commands);
    
    cmd_lists.at(cmd_list_count++) = _scene->record_gpu_manifest_update({
        .uploaded_meshes = asset_data_upload_info.uploaded_meshes,
        .uploaded_textures = asset_data_upload_info.uploaded_textures,
    });
    cmd_lists.at(cmd_list_count++) = _scene->create_mesh_acceleration_structures();
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
    _ui_engine->main_update(*_gpu_context, *_renderer->render_context, *_scene, app_state, *_window);
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

    // ===== Input Handling =====
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