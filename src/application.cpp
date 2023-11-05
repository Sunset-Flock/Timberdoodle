#include "application.hpp"
#include "ui.hpp"
#include "fmt/format.h"

void CameraController::process_input(Window &window, f32 dt)
{
    f32 speed = window.key_pressed(GLFW_KEY_LEFT_SHIFT) ? translationSpeed * 4.0f : translationSpeed;
    speed = window.key_pressed(GLFW_KEY_LEFT_CONTROL) ? speed * 0.25f : speed;

    if (window.is_focused())
    {
        if (window.key_just_pressed(GLFW_KEY_ESCAPE))
        {
            if (window.is_cursor_captured())
            {
                window.release_cursor();
            }
            else
            {
                window.capture_cursor();
            }
        }
    }
    else if (window.is_cursor_captured())
    {
        window.release_cursor();
    }

    auto cameraSwaySpeed = this->cameraSwaySpeed;
    if (window.key_pressed(GLFW_KEY_C))
    {
        cameraSwaySpeed *= 0.25;
        bZoom = true;
    }
    else
    {
        bZoom = false;
    }

    glm::vec3 right = glm::cross(forward, up);
    glm::vec3 fake_up = glm::cross(right, forward);
    if (window.is_cursor_captured())
    {
        if (window.key_pressed(GLFW_KEY_W))
        {
            position += forward * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_S))
        {
            position -= forward * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_A))
        {
            position -= glm::normalize(glm::cross(forward, up)) * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_D))
        {
            position += glm::normalize(glm::cross(forward, up)) * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_SPACE))
        {
            position += fake_up * speed * dt;
        }
        if (window.key_pressed(GLFW_KEY_LEFT_ALT))
        {
            position -= fake_up * speed * dt;
        }
        pitch += window.get_cursor_change_y() * cameraSwaySpeed;
        pitch = std::clamp(pitch, -85.0f, 85.0f);
        yaw += window.get_cursor_change_x() * cameraSwaySpeed;
    }
    forward.x = -glm::cos(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.y = glm::sin(glm::radians(yaw - 90.0f)) * glm::cos(glm::radians(pitch));
    forward.z = -glm::sin(glm::radians(pitch));
}

void CameraController::update_matrices(Window &window)
{
    auto fov = this->fov;
    if (bZoom)
    {
        fov *= 0.25f;
    }
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
    glm::mat4 prespective = inf_depth_reverse_z_perspective(glm::radians(fov), f32(window.get_width()) / f32(window.get_height()), near);
    prespective[1][1] *= -1.0f;
    this->cam_info.proj = prespective;
    this->cam_info.view = glm::lookAt(position, position + forward, up);
    this->cam_info.vp = this->cam_info.proj * this->cam_info.view;
    this->cam_info.pos = this->position;
    this->cam_info.up = this->up;
    glm::vec3 ws_ndc_corners[2][2][2];
    glm::mat4 inv_view_proj = glm::inverse(this->cam_info.proj * this->cam_info.view);
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
    this->cam_info.camera_near_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[0][1][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0]));
    this->cam_info.camera_right_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
    this->cam_info.camera_left_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
    this->cam_info.camera_top_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
    this->cam_info.camera_bottom_plane_normal = glm::normalize(glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));
    int i = 0;
}

Application::Application()
{
    _window = std::make_unique<Window>(1920, 1080, "Sandbox");

    _gpu_context = std::make_unique<GPUContext>(*_window);

    _scene = std::make_unique<Scene>(_gpu_context->device);
    // TODO(ui): DO NOT ALWAYS JUST LOAD THIS UNCONDITIONALLY!
    // TODO(ui): ADD UI FOR LOADING IN THE EDITOR!
    std::filesystem::path const DEFAULT_HARDCODED_PATH = ".\\assets";
    std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro_gltf\\bistro.gltf";
    auto const result = _scene->load_manifest_from_gltf(DEFAULT_HARDCODED_PATH, DEFAULT_HARDCODED_FILE);
    if (Scene::LoadManifestErrorCode const *err = std::get_if<Scene::LoadManifestErrorCode>(&result))
    {
        fmt::println("[WARN][Application::Application()] Loading \"{}\" Error: {}",
                     (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string(),
                     Scene::to_string(*err));
    }
    else
    {
        auto const r_id = std::get<RenderEntityId>(result);
        RenderEntity& r_ent = *_scene->_render_entities.slot(r_id);
        r_ent.transform = glm::mat4x3(
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 0.0f)
        ) * 100.0f;
        fmt::println("[INFO][Application::Application()] Loading \"{}\" Success",
                     (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string());
    }
    auto scene_commands = _scene->record_gpu_manifest_update();

    _asset_manager = std::make_unique<AssetProcessor>(_gpu_context->device);
    _asset_manager->load_all(*_scene);
    auto exc_cmd_list = _asset_manager->record_gpu_load_processing_commands();
    _gpu_context->device.submit_commands({.command_lists = std::array{std::move(scene_commands), std::move(exc_cmd_list)}});
    _gpu_context->device.wait_idle();

    _ui_engine = std::make_unique<UIEngine>(*_window);

    _renderer = std::make_unique<Renderer>(_window.get(), _gpu_context.get(), _scene.get(), _asset_manager.get());

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
        daxa_i32vec2 new_window_size;
        glfwGetWindowSize(this->_window->glfw_handle, &new_window_size.x, &new_window_size.y);
        if (this->_window->size.x != new_window_size.x || _window->size.y != new_window_size.y)
        {
            this->_window->size = new_window_size;
            _renderer->window_resized();
        }
        update();
        _renderer->render_frame(this->camera_controller.cam_info, this->observer_camera_controller.cam_info, delta_time);
    }
    return 0;
}

void Application::update()
{
    if (_window->size.x == 0 || _window->size.y == 0)
    {
        return;
    }
    // _ui_engine.main_update(_gpu_context->settings);
    if (control_observer)
    {
        observer_camera_controller.process_input(*_window, this->delta_time);
        observer_camera_controller.update_matrices(*_window);
    }
    else
    {
        camera_controller.process_input(*_window, this->delta_time);
        camera_controller.update_matrices(*_window);
    }
    if (_window->key_just_pressed(GLFW_KEY_H))
    {
        DEBUG_MSG(fmt::format("switched enable_observer from {} to {}",_renderer->context->settings.enable_observer, !(_renderer->context->settings.enable_observer)));
        _renderer->context->settings.enable_observer = !_renderer->context->settings.enable_observer;
    }
    if (_window->key_just_pressed(GLFW_KEY_J))
    {
        DEBUG_MSG(fmt::format("switched control_observer from {} to {}",control_observer, !(control_observer)));
        control_observer = !control_observer;
    }
    if (_window->key_just_pressed(GLFW_KEY_K))
    {
        DEBUG_MSG("reset observer");
        control_observer = false;
        _renderer->context->settings.enable_observer = false;
        observer_camera_controller = camera_controller;
    }
#if COMPILE_IN_MESH_SHADER
    if (_window->key_just_pressed(GLFW_KEY_M))
    {
        DEBUG_MSG(fmt::format("switched enable_mesh_shader from {} to {}",_renderer->context->settings.enable_mesh_shader, !(_renderer->context->settings.enable_mesh_shader)));
        _renderer->context->settings.enable_mesh_shader = !_renderer->context->settings.enable_mesh_shader;
    }
#endif
    if (_window->key_just_pressed(GLFW_KEY_O))
    {
        DEBUG_MSG(fmt::format("switched observer_show_pass from {} to {}", _renderer->context->settings.observer_show_pass, ((_renderer->context->settings.observer_show_pass + 1) % 3)));
        _renderer->context->settings.observer_show_pass = (_renderer->context->settings.observer_show_pass + 1) % 3;
    }
}

Application::~Application()
{
}