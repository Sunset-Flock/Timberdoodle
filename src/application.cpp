#include "application.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <fstream>

#include <intrin.h>

void CameraController::process_input(Window & window, f32 dt)
{
    f32 speed = window.key_pressed(GLFW_KEY_LEFT_SHIFT) ? translationSpeed * 4.0f : translationSpeed;
    speed = window.key_pressed(GLFW_KEY_LEFT_CONTROL) ? speed * 0.25f : speed;

    if (window.is_focused())
    {
        if (window.key_just_pressed(GLFW_KEY_ESCAPE))
        {
            if (window.is_cursor_captured()) { window.release_cursor(); }
            else { window.capture_cursor(); }
        }
    }
    else if (window.is_cursor_captured()) { window.release_cursor(); }

    auto cameraSwaySpeed = this->cameraSwaySpeed;
    if (window.key_pressed(GLFW_KEY_C))
    {
        cameraSwaySpeed *= 0.25;
        bZoom = true;
    }
    else { bZoom = false; }

    glm::vec3 right = glm::cross(forward, up);
    glm::vec3 fake_up = glm::cross(right, forward);
    if (window.is_cursor_captured())
    {
        if (window.key_pressed(GLFW_KEY_W)) { position += forward * speed * dt; }
        if (window.key_pressed(GLFW_KEY_S)) { position -= forward * speed * dt; }
        if (window.key_pressed(GLFW_KEY_A)) { position -= glm::normalize(glm::cross(forward, up)) * speed * dt; }
        if (window.key_pressed(GLFW_KEY_D)) { position += glm::normalize(glm::cross(forward, up)) * speed * dt; }
        if (window.key_pressed(GLFW_KEY_SPACE)) { position += fake_up * speed * dt; }
        if (window.key_pressed(GLFW_KEY_LEFT_ALT)) { position -= fake_up * speed * dt; }
        pitch += window.get_cursor_change_y() * cameraSwaySpeed;
        pitch = std::clamp(pitch, -85.0f, 85.0f);
        yaw += window.get_cursor_change_x() * cameraSwaySpeed;
    }
    forward.x = -glm::cos(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.y = glm::sin(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.z = -glm::sin(glm::radians(pitch));
}

auto CameraController::make_camera_info(Settings const & settings) const -> CameraInfo
{
    auto fov = this->fov;
    if (bZoom) { fov *= 0.25f; }
    auto inf_depth_reverse_z_perspective = [](auto fov_rads, auto aspect, auto zNear)
    {
        assert(abs(aspect - std::numeric_limits<f32>::epsilon()) > 0.0f);

        f32 const tanHalfFovy = 1.0f / std::tan(fov_rads * 0.5f);

        glm::mat4x4 ret(0.0f);
        ret[0][0] = tanHalfFovy / aspect;
        ret[1][1] = tanHalfFovy;
        ret[2][2] = 0.0f;
        ret[2][3] = -1.0f;
        ret[3][2] = zNear;
        return ret;
    };
    glm::mat4 prespective =
        inf_depth_reverse_z_perspective(glm::radians(fov), f32(settings.render_target_size.x) / f32(settings.render_target_size.y), near);
    prespective[1][1] *= -1.0f;
    CameraInfo ret = {};
    ret.proj = prespective;
    ret.inv_proj = glm::inverse(prespective);
    ret.view = glm::lookAt(position, position + forward, up);
    ret.inv_view = glm::inverse(ret.view);
    ret.view_proj = ret.proj * ret.view;
    ret.inv_view_proj = glm::inverse(ret.view_proj);
    ret.position = this->position;
    ret.up = this->up;
    glm::vec3 ws_ndc_corners[2][2][2];
    glm::mat4 inv_view_proj = glm::inverse(ret.proj * ret.view);
    for (u32 z = 0; z < 2; ++z)
    {
        for (u32 y = 0; y < 2; ++y)
        {
            for (u32 x = 0; x < 2; ++x)
            {
                glm::vec3 corner = glm::vec3((glm::vec2(x, y) - 0.5f) * 2.0f, 1.0f - z * 0.5f);
                glm::vec4 proj_corner = inv_view_proj * glm::vec4(corner, 1);
                ws_ndc_corners[x][y][z] = glm::vec3(proj_corner) / proj_corner.w;
            }
        }
    }
    ret.near_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0]));
    ret.right_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
    ret.left_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
    ret.top_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
    ret.bottom_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));
    int i = 0;
    ret.screen_size = { settings.render_target_size.x, settings.render_target_size.y };
    ret.inv_screen_size = {
        1.0f / static_cast<f32>(settings.render_target_size.x),
        1.0f / static_cast<f32>(settings.render_target_size.y),
    };
    ret.near_plane = this->near;
    return ret;
}

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

    struct CompPipelinesTask : Task
    {
        Renderer * renderer = {};
        CompPipelinesTask(Renderer * renderer)
            : renderer{renderer} { chunk_count = 1; }

        virtual void callback(u32 chunk_index, u32 thread_index) override
        {
            // TODO: hook up parameters.
            renderer->compile_pipelines(false, false);
        }
    };

    auto comp_pipelines_task = std::make_shared<CompPipelinesTask>(_renderer.get());

    _threadpool->async_dispatch(comp_pipelines_task);
    _threadpool->block_on(comp_pipelines_task);

    // TODO(ui): DO NOT ALWAYS JUST LOAD THIS UNCONDITIONALLY!
    // TODO(ui): ADD UI FOR LOADING IN THE EDITOR!
    std::filesystem::path const DEFAULT_HARDCODED_PATH = ".\\assets";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "suzanne\\suzanne.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "old_sponza\\old_sponza.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "new_sponza\\NewSponza_Main_glTF_002.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro\\bistro.gltf";
    std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro_compressed\\bistro_c.gltf";
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
        if(_window->size.x != 0 && _window->size.y != 0) 
        {
            update();
            _renderer->render_frame(
                this->camera_controller, 
                this->observer_camera_controller, 
                delta_time,
                this->_scene->_scene_draw);
        }
        _gpu_context->device.collect_garbage();
    }
    return 0;
}

void Application::update()
{
    auto asset_data_upload_info = _asset_manager->record_gpu_load_processing_commands();
    auto manifest_update_commands = _scene->record_gpu_manifest_update({
        .uploaded_meshes = asset_data_upload_info.uploaded_meshes,
        .uploaded_textures = asset_data_upload_info.uploaded_textures,
    });
    auto cmd_lists = std::array{std::move(asset_data_upload_info.upload_commands), std::move(manifest_update_commands)};
    _gpu_context->device.submit_commands({.command_lists = cmd_lists});

    bool reset_observer = false;
    if (_window->size.x == 0 || _window->size.y == 0) { return; }
    _ui_engine->main_update(*_renderer->render_context, *_scene);
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
                ImGui::Checkbox("control observer   (J)", &control_observer);
                reset_observer = reset_observer || (ImGui::Button("reset observer     (K)"));
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
                    reinterpret_cast<f32*>(&_renderer->context->shader_debug_context.shader_debug_input.debug_fvec4),
                    _ui_engine->debug_f32vec4_drag_speed);
                ImGui::DragInt4(
                    "debug i32vec4", 
                    reinterpret_cast<i32*>(&_renderer->context->shader_debug_context.shader_debug_input.debug_ivec4));
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
                ImGui::Image(debug_lens_image_view_id, ImVec2(width,width));
            }
            ImGui::End();
        }
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
        if constexpr (requires(decltype(val) x){x.z;}) val.z = json[name]["z"];
        if constexpr (requires(decltype(val) x){x.w;}) val.w = json[name]["w"];
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
        daxa_f32(glm::cos(glm::radians(sun_angle.y)))
    };
    settings.sun_brightness = 10.0f;
}

Application::~Application()
{
}