#include "application.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <fstream>

#include <intrin.h>

Application::Application()
{
    _threadpool = std::make_unique<ThreadPool>(7);
    _window = std::make_unique<Window>(1024, 1024, "Sandbox");
    _gpu_context = std::make_unique<GPUContext>(*_window);
    _scene = std::make_unique<Scene>(_gpu_context->device);
    _asset_manager = std::make_unique<AssetProcessor>(_gpu_context->device);
    _ui_engine = std::make_unique<UIEngine>(*_window, *_asset_manager, _gpu_context.get());

    _renderer = std::make_unique<Renderer>(_window.get(), _gpu_context.get(), _scene.get(), _asset_manager.get(), &_ui_engine->imgui_renderer);
    // Renderer needs these to be loaded to know what size the look up tables have to be
    std::filesystem::path const DEFAULT_SKY_SETTINGS_PATH = "settings\\sky\\default.json";
    load_sky_settings(DEFAULT_SKY_SETTINGS_PATH, _renderer->render_context->render_data.sky_settings);

    std::vector<AnimationKeyframe> keyframes = {
        AnimationKeyframe{
            .start_rotation = {-0.30450, 0.275662, 0.611860, 0.6759723},
            .end_rotation = {0.141418, -0.15007, 0.712131, 0.6710799},
            .start_position = {2.86, 25.80, 12.00},
            .first_control_point = {-3.45, 28.96, 12.04},
            .second_control_point = {-13.84, 9.95, 1.74},
            .end_position = {-14.07, 2.20, 2.12},
            .transition_time = 4.0f,
        },
        AnimationKeyframe{
            .start_rotation = {0.141418, -0.15007, 0.712131, 0.6710799},
            .end_rotation = {0.38337, -0.382710, 0.593852, 0.5948849},
            .start_position = {-14.07, 2.20, 2.12},
            .first_control_point = {-14.31, -5.54, 2.48},
            .second_control_point = {-11.46, -7.74, 1.13},
            .end_position = {-8.88, -9.17, 1.41},
            .transition_time = 3.0f,
        },
        AnimationKeyframe{
            .start_rotation = {0.38337, -0.382710, 0.593852, 0.5948849},
            .end_rotation = {0.341726, -0.374238, 0.636601, 0.581297},
            .start_position = {-8.88, -9.17, 1.41},
            .first_control_point = {-6.30, -10.59, 0.94},
            .second_control_point = {50.59, -30.63, 1.80},
            .end_position = {54.51, -35.18, 1.14},
            .transition_time = 3.0f,
        },
        AnimationKeyframe{
            .start_rotation = {0.341726, -0.374238, 0.636601, 0.581297},
            .end_rotation = {-0.536079, 0.5606119, -0.4617242, -0.4370127},
            .start_position = {54.51, -35.18, 1.14},
            .first_control_point = {58.43, -39.73, 0.47},
            .second_control_point = {54.23, -46.38, 3.83},
            .end_position = {64.29, -52.73, 3.83},
            .transition_time = 3.0f,
        },
        AnimationKeyframe{
            .start_rotation = {-0.536079, 0.5606119, -0.4617242, -0.4370127},
            .end_rotation = {0.7877, -0.4515774, -0.208418, -0.363549},
            .start_position = {64.29, -52.73, 3.83},
            .first_control_point = {74.34, -58.95, 3.83},
            .second_control_point = {80.60, -53.08, 8.85},
            .end_position = {84.23, -45.45, 10.64},
            .transition_time = 3.0f,
        },
        AnimationKeyframe{
            .start_rotation = {0.7877, -0.4515774, -0.208418, -0.363549},
            .end_rotation = {-0.1558, 0.0928755, 0.503549, 0.8447122},
            .start_position = {84.23, -45.45, 10.64},
            .first_control_point = {87.93, -37.94, 12.49},
            .second_control_point = {84.22, -26.02, 14.57},
            .end_position = {72.97, -25.84, 10.65},
            .transition_time = 3.0f,
        },
        AnimationKeyframe{
            .start_rotation = {-0.1558, 0.0928755, 0.503549, 0.8447122},
            .end_rotation = {0.604737, -0.585519, -0.3755327, -0.3878583},
            .start_position = {72.97, -25.84, 10.65},
            .first_control_point = {61.71, -25.67, 6.73},
            .second_control_point = {62.31, -34.58, 4.07},
            .end_position = {51.06, -30.80, 3.04},
            .transition_time = 3.0f,
        },
        AnimationKeyframe{
            .start_rotation = {0.604737, -0.585519, -0.3755327, -0.3878583},
            .end_rotation = {-0.646975, 0.756841, -0.0705465, -0.0603057},
            .start_position = {51.06, -30.80, 3.04},
            .first_control_point = {39.80, -27.02, 2.00},
            .second_control_point = {-0.32, -14.83, 3.63},
            .end_position = {-7.36, -5.53, 3.97},
            .transition_time = 3.0f,
        },
        AnimationKeyframe{
            .start_rotation = {-0.646975, 0.756841, -0.0705465, -0.0603057},
            .end_rotation = {-0.30450, 0.275662, 0.611860, 0.6759723},
            .start_position = {-7.36, -5.53, 3.97},
            .first_control_point = {-14.39, 3.76, 4.31},
            .second_control_point = {9.44, 22.51, 11.27},
            .end_position = {2.86, 25.80, 12.00},
            .transition_time = 3.0f,
        },
    };
    cinematic_camera.update_keyframes(std::move(keyframes));

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
    std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro_compressed\\bistro_c.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro_fix_ball_compressed\\bistro_fix_ball_c.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "battle_scene_compressed\\battle_scene_c.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "cube.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "TestWorld\\TestWorld.gltf";

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
                    dynamic_ball = child.value();
                    break;
                }
                child = child_node.next_sibling;
            }
        }

        DEBUG_MSG(fmt::format("[INFO][Application::Application()] Loading \"{}\" Success",
            (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string()));
    }

    last_time_point = std::chrono::steady_clock::now();
}
using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;

auto Application::run() -> i32
{
    while (keep_running)
    {
        auto new_time_point = std::chrono::steady_clock::now();
        this->delta_time = std::chrono::duration_cast<FpMilliseconds>(new_time_point - this->last_time_point).count() * 0.001f;
        this->last_time_point = new_time_point;
        _window->update(delta_time);
        keep_running &= !static_cast<bool>(glfwWindowShouldClose(_window->glfw_handle));
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
            auto const camera_info = use_preset_camera ? cinematic_camera.make_camera_info(_renderer->render_context->render_data.settings) : camera_controller.make_camera_info(_renderer->render_context->render_data.settings);

            _renderer->render_frame(
                camera_info,
                observer_camera_controller.make_camera_info(_renderer->render_context->render_data.settings),
                delta_time,
                this->_scene->_scene_draw);
        }
        _gpu_context->device.collect_garbage();
    }
    return 0;
}

void Application::update()
{
    auto * dynamic_ball_ent = _scene->_render_entities.slot(dynamic_ball);
    auto prev_transform = glm::mat4(
        glm::vec4(dynamic_ball_ent->transform[0], 0.0f),
        glm::vec4(dynamic_ball_ent->transform[1], 0.0f),
        glm::vec4(dynamic_ball_ent->transform[2], 0.0f),
        glm::vec4(dynamic_ball_ent->transform[3], 1.0f));
    
    static f32 total_time = 0.0f;
    total_time += delta_time;
    auto new_position = f32vec4{
        std::sin(total_time) * 100.0f,
        std::cos(total_time) * 100.0f,
        prev_transform[3].z,
        1.0f
    };
    auto curr_transform = prev_transform;
    curr_transform[3] = new_position;

    dynamic_ball_ent->transform = curr_transform;
    _scene->_modified_render_entities.push_back({dynamic_ball, prev_transform, curr_transform});

    auto asset_data_upload_info = _asset_manager->record_gpu_load_processing_commands();
    auto manifest_update_commands = _scene->record_gpu_manifest_update({
        .uploaded_meshes = asset_data_upload_info.uploaded_meshes,
        .uploaded_textures = asset_data_upload_info.uploaded_textures,
    });
    auto cmd_lists = std::array{std::move(asset_data_upload_info.upload_commands), std::move(manifest_update_commands)};
    _gpu_context->device.submit_commands({.command_lists = cmd_lists});

    bool reset_observer = false;
    if (_window->size.x == 0 || _window->size.y == 0) {
        return; }
    _ui_engine->main_update(*_renderer->render_context, *_scene);
    if(use_preset_camera)
    {
        cinematic_camera.process_input(*_window, this->delta_time);
    }
    if (control_observer)
    {
        observer_camera_controller.process_input(*_window, this->delta_time);
    }
    else
    {
        camera_controller.process_input(*_window, this->delta_time);
    }
    if (_ui_engine->shader_debug_menu)
    {
        if (ImGui::Begin("Shader Debug Menu", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::SeparatorText("Observer Camera");
            {
                IMGUI_UINT_CHECKBOX2("draw from observer (H)", _renderer->render_context->render_data.settings.draw_from_observer);
                ImGui::Checkbox("use preset camera", &use_preset_camera);
                ImGui::Checkbox("control observer   (J)", &control_observer);
                auto const view_quat = glm::quat_cast(observer_camera_controller.make_camera_info(_renderer->render_context->render_data.settings).view);
                ImGui::Text("%s", fmt::format("observer view quat {} {} {} {}", view_quat.w, view_quat.x, view_quat.y, view_quat.z).c_str());
                ImGui::BeginDisabled(!use_preset_camera);
                ImGui::Checkbox("Override keyframe (I)", &cinematic_camera.override_keyframe);
                ImGui::EndDisabled();
                cinematic_camera.override_keyframe = _window->key_just_pressed(GLFW_KEY_I) ? !cinematic_camera.override_keyframe : cinematic_camera.override_keyframe;
                cinematic_camera.override_keyframe &= use_preset_camera;
                ImGui::BeginDisabled(!cinematic_camera.override_keyframe);
                i32 current_keyframe = cinematic_camera.current_keyframe_index;
                f32 keyframe_progress = cinematic_camera.current_keyframe_time / cinematic_camera.path_keyframes.at(current_keyframe).transition_time;
                ImGui::SliderInt("keyframe", &current_keyframe, 0, cinematic_camera.path_keyframes.size() - 1);
                ImGui::SliderFloat("keyframe progress", &keyframe_progress, 0.0f, 1.0f);
                if (cinematic_camera.override_keyframe) { cinematic_camera.set_keyframe(current_keyframe, keyframe_progress); }
                ImGui::EndDisabled();
                reset_observer = reset_observer || (ImGui::Button("reset observer     (K)"));
                if (ImGui::Button("snap observer to cinematic"))
                {
                    observer_camera_controller.position = cinematic_camera.position;
                }
                std::array<char const * const, 3> modes = {
                    "redraw meshlets visible last frame",
                    "redraw meshlet post cull",
                    "redraw all drawn meshlets",
                };
                ImGui::Combo("observer draw pass mode", &_renderer->render_context->render_data.settings.observer_show_pass, modes.data(), modes.size());
                if (_window->key_just_pressed(GLFW_KEY_H))
                {
                    _renderer->render_context->render_data.settings.draw_from_observer = !_renderer->render_context->render_data.settings.draw_from_observer;
                }
                if (_window->key_just_pressed(GLFW_KEY_J))
                {
                    control_observer = !control_observer;
                }
                if (_window->key_just_pressed(GLFW_KEY_K))
                {
                    reset_observer = true;
                }
                if (_window->key_just_pressed(GLFW_KEY_O))
                {
                    DEBUG_MSG(fmt::format("switched observer_show_pass from {} to {}", _renderer->render_context->render_data.settings.observer_show_pass,
                        ((_renderer->render_context->render_data.settings.observer_show_pass + 1) % 3)));
                    _renderer->render_context->render_data.settings.observer_show_pass = (_renderer->render_context->render_data.settings.observer_show_pass + 1) % 3;
                }
            }
            ImGui::SeparatorText("Debug Shader Interface");
            {
                ImGui::InputFloat("debug f32vec4 drag speed", &_ui_engine->debug_f32vec4_drag_speed);
                ImGui::DragFloat4(
                    "debug f32vec4",
                    reinterpret_cast<f32 *>(&_renderer->context->shader_debug_context.shader_debug_input.debug_fvec4),
                    _ui_engine->debug_f32vec4_drag_speed);
                ImGui::DragInt4(
                    "debug i32vec4",
                    reinterpret_cast<i32 *>(&_renderer->context->shader_debug_context.shader_debug_input.debug_ivec4));
                ImGui::Text(
                    "out debug f32vec4: (%f,%f,%f,%f)",
                    _renderer->context->shader_debug_context.shader_debug_output.debug_fvec4.x,
                    _renderer->context->shader_debug_context.shader_debug_output.debug_fvec4.y,
                    _renderer->context->shader_debug_context.shader_debug_output.debug_fvec4.z,
                    _renderer->context->shader_debug_context.shader_debug_output.debug_fvec4.w);
                ImGui::Text(
                    "out debug i32vec4: (%i,%i,%i,%i)",
                    _renderer->context->shader_debug_context.shader_debug_output.debug_ivec4.x,
                    _renderer->context->shader_debug_context.shader_debug_output.debug_ivec4.y,
                    _renderer->context->shader_debug_context.shader_debug_output.debug_ivec4.z,
                    _renderer->context->shader_debug_context.shader_debug_output.debug_ivec4.w);
                if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->button_just_pressed(GLFW_MOUSE_BUTTON_1))
                {
                    _renderer->context->shader_debug_context.detector_window_position = {
                        _window->get_cursor_x(),
                        _window->get_cursor_y(),
                    };
                }
                if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_LEFT))
                {
                    _renderer->context->shader_debug_context.detector_window_position.x -= 1;
                }
                if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_RIGHT))
                {
                    _renderer->context->shader_debug_context.detector_window_position.x += 1;
                }
                if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_UP))
                {
                    _renderer->context->shader_debug_context.detector_window_position.y -= 1;
                }
                if (_window->key_pressed(GLFW_KEY_LEFT_ALT) && _window->key_just_pressed(GLFW_KEY_DOWN))
                {
                    _renderer->context->shader_debug_context.detector_window_position.y += 1;
                }
            }
            ImGui::SeparatorText("Debug Shader Lens");
            {
                ImGui::Text("Press ALT + LEFT_CLICK to set the detector to the cursor position");
                ImGui::Text("Press ALT + Keyboard arrow keys to move detector");
                ImGui::Checkbox("draw_magnified_area_rect", &_renderer->context->shader_debug_context.draw_magnified_area_rect);
                ImGui::InputInt("detector window size", &_renderer->context->shader_debug_context.detector_window_size, 2);
                ImGui::Text(
                    "detector texel position: (%i,%i)",
                    _renderer->context->shader_debug_context.detector_window_position.x,
                    _renderer->context->shader_debug_context.detector_window_position.y);
                ImGui::Text(
                    "detector center value: (%f,%f,%f,%f)",
                    _renderer->context->shader_debug_context.shader_debug_output.texel_detector_center_value.x,
                    _renderer->context->shader_debug_context.shader_debug_output.texel_detector_center_value.y,
                    _renderer->context->shader_debug_context.shader_debug_output.texel_detector_center_value.z,
                    _renderer->context->shader_debug_context.shader_debug_output.texel_detector_center_value.w);
                auto debug_lens_image_view_id = _ui_engine->imgui_renderer.create_texture_id({
                    .image_view_id = _renderer->context->shader_debug_context.debug_lens_image.default_view(),
                    .sampler_id = std::bit_cast<daxa::SamplerId>(_renderer->render_context->render_data.samplers.nearest_clamp),
                });
                auto const width = ImGui::GetContentRegionMax().x;
                ImGui::Image(debug_lens_image_view_id, ImVec2(width, width));
            }
        }
        ImGui::End();
    }
    if (reset_observer)
    {
        control_observer = false;
        _renderer->render_context->render_data.settings.draw_from_observer = static_cast<u32>(false);
        observer_camera_controller = camera_controller;
    }
    ImGui::Render();
}

void Application::load_sky_settings(std::filesystem::path const path_to_settings, SkySettings & settings)
{
    auto json = nlohmann::json::parse(std::ifstream(path_to_settings));
    auto read_val = [&json](auto const name, auto & val)
    {
        val = json[name];
    };

    auto read_vec = [&json](auto const name, auto & val)
    {
        val.x = json[name]["x"];
        val.y = json[name]["y"];
        if constexpr (requires(decltype(val) x) { x.z; }) val.z = json[name]["z"];
        if constexpr (requires(decltype(val) x) { x.w; }) val.w = json[name]["w"];
    };

    auto read_density_profile_layer = [&json](auto const name, auto const layer, DensityProfileLayer & val)
    {
        val.layer_width = json[name][layer]["layer_width"];
        val.exp_term = json[name][layer]["exp_term"];
        val.exp_scale = json[name][layer]["exp_scale"];
        val.lin_term = json[name][layer]["lin_term"];
        val.const_term = json[name][layer]["const_term"];
    };
    read_vec("transmittance_dimensions", settings.transmittance_dimensions);
    read_vec("multiscattering_dimensions", settings.multiscattering_dimensions);
    read_vec("sky_dimensions", settings.sky_dimensions);

    f32vec2 sun_angle = {};
    read_vec("sun_angle", sun_angle);

    read_val("atmosphere_bottom", settings.atmosphere_bottom);
    read_val("atmosphere_top", settings.atmosphere_top);

    // Mie
    read_vec("mie_scattering", settings.mie_scattering);
    read_vec("mie_extinction", settings.mie_extinction);
    read_val("mie_scale_height", settings.mie_scale_height);
    read_val("mie_phase_function_g", settings.mie_phase_function_g);
    read_density_profile_layer("mie_density", 0, settings.mie_density[0]);
    read_density_profile_layer("mie_density", 1, settings.mie_density[1]);

    // Rayleigh
    read_vec("rayleigh_scattering", settings.rayleigh_scattering);
    read_val("rayleigh_scale_height", settings.rayleigh_scale_height);
    read_density_profile_layer("rayleigh_density", 0, settings.rayleigh_density[0]);
    read_density_profile_layer("rayleigh_density", 1, settings.rayleigh_density[1]);

    // Absorption
    read_vec("absorption_extinction", settings.absorption_extinction);
    read_density_profile_layer("absorption_density", 0, settings.absorption_density[0]);
    read_density_profile_layer("absorption_density", 1, settings.absorption_density[1]);

    settings.sun_direction =
        {
            daxa_f32(glm::cos(glm::radians(sun_angle.x)) * glm::sin(glm::radians(sun_angle.y))),
            daxa_f32(glm::sin(glm::radians(sun_angle.x)) * glm::sin(glm::radians(sun_angle.y))),
            daxa_f32(glm::cos(glm::radians(sun_angle.y)))};
    settings.sun_brightness = 10.0f;
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