#include "application.hpp"
#include <fmt/core.h>
#include <fmt/format.h>

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

void CameraController::update_matrices(Window & window)
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
        inf_depth_reverse_z_perspective(glm::radians(fov), f32(window.get_width()) / f32(window.get_height()), near);
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
    this->cam_info.camera_near_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0]));
    this->cam_info.camera_right_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][1][0] - ws_ndc_corners[1][0][0], ws_ndc_corners[1][0][1] - ws_ndc_corners[1][0][0]));
    this->cam_info.camera_left_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][0][1], ws_ndc_corners[0][0][0] - ws_ndc_corners[0][0][1]));
    this->cam_info.camera_top_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[1][0][0] - ws_ndc_corners[0][0][0], ws_ndc_corners[0][0][1] - ws_ndc_corners[0][0][0]));
    this->cam_info.camera_bottom_plane_normal = glm::normalize(
        glm::cross(ws_ndc_corners[0][1][1] - ws_ndc_corners[0][1][0], ws_ndc_corners[1][1][0] - ws_ndc_corners[0][1][0]));
    int i = 0;
}

#include <cstdlib>
template <typename Clock = std::chrono::high_resolution_clock>
struct Stopwatch
{
    public:
    Stopwatch() : start_point(Clock::now()) {}

    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsed_time() const
    {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }

    private:
    typename Clock::time_point const start_point;
};

u32 expensive_op(u32 value)
{
    return u32(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(std::sqrt(float(value))))))))))))));
}

using PreciseStopwatch = Stopwatch<>;
using SystemStopwatch = Stopwatch<std::chrono::system_clock>;
using MonotonicStopwatch = Stopwatch<std::chrono::steady_clock>;
bool test(std::unique_ptr<ThreadPool> const & tp)
{

    /// TESTING +=====================================
    struct SumTask : Task
    {
        u32 const DATA_SIZE = 16'000'000u;
        u32 const CHUNK_COUNT = 16'00u;
        u32 const CHUNK_SIZE = DATA_SIZE / CHUNK_COUNT;
        std::vector<u32> data = {};
        std::atomic_uint32_t result = 0;
        SumTask()
        {
            data = std::vector<u32>(DATA_SIZE);
            for (u32 i = 0; i < DATA_SIZE; i++)
            {
                data.at(i) = rand() % 100;
            }
            chunk_count = CHUNK_COUNT;
        };

        virtual void callback(u32 chunk_index, u32 thread_index) override
        {
            u32 const start_index = chunk_index * CHUNK_SIZE;
            u32 const end_index = (chunk_index + 1) * CHUNK_SIZE;
            u32 local_result = 0;
            for (u32 i = start_index; i < end_index; i += 1)
            {
                local_result += expensive_op(data.at(i));
            }
            result += local_result;
        };
    };


    auto task = std::make_shared<SumTask>();
    PreciseStopwatch stopwatch = {};
    u32 mt_start_time = stopwatch.elapsed_time<u32, std::chrono::microseconds>();
    tp->blocking_dispatch(task);
    u32 mt_end_time = stopwatch.elapsed_time<u32, std::chrono::microseconds>();
    fmt::println("Blocking dispatch took {}us", mt_end_time - mt_start_time);

    u32 single_threaded_result = 0;
    u32 st_start_time = stopwatch.elapsed_time<u32, std::chrono::microseconds>();
    for (u32 i = 0; i < task->DATA_SIZE; i++)
    {
        single_threaded_result += expensive_op(task->data.at(i));
    }
    u32 st_end_time = stopwatch.elapsed_time<u32, std::chrono::microseconds>();
    fmt::println("Single treaded dispatch took {}us", st_end_time - st_start_time);

    if (single_threaded_result != task->result) { 
        fmt::println("Single threaded and multi threaded results don't match");
        return false;
    }
    else
    { 
        fmt::println("Single threaded result matches multithreded {} with multithreaded being {}x faster",
            task->result.load(), 
            f32(st_end_time - st_start_time)/f32(mt_end_time - mt_start_time)
        ); 
        return true;
    }
}

Application::Application()
{

    _threadpool = std::make_unique<ThreadPool>(8);
    bool tests_passed = true;
    for(u32 test_iteration = 0; test_iteration < 1000; test_iteration++)
    {
        fmt::println("Running test iteration {}", test_iteration);
        if(!test(_threadpool))
        {
            fmt::println("TEST FAILED on iteration {}", test_iteration);
            tests_passed = false;
            break;
        }
    }
    if(!tests_passed)
    {
        fmt::println("TESTS DID NOT PASS");
    }

    _window = std::make_unique<Window>(1920, 1080, "Sandbox");

    _gpu_context = std::make_unique<GPUContext>(*_window);

    _scene = std::make_unique<Scene>(_gpu_context->device);
    // TODO(ui): DO NOT ALWAYS JUST LOAD THIS UNCONDITIONALLY!
    // TODO(ui): ADD UI FOR LOADING IN THE EDITOR!
    std::filesystem::path const DEFAULT_HARDCODED_PATH = ".\\assets";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "suzanne\\suzanne.gltf";
    // std::filesystem::path const DEFAULT_HARDCODED_FILE = "bistro_gltf\\bistro.gltf";
    std::filesystem::path const DEFAULT_HARDCODED_FILE = "new_sponza\\NewSponza_Main_glTF_002.gltf";
    auto const result = _scene->load_manifest_from_gltf(DEFAULT_HARDCODED_PATH, DEFAULT_HARDCODED_FILE);
    if (Scene::LoadManifestErrorCode const * err = std::get_if<Scene::LoadManifestErrorCode>(&result))
    {
        DEBUG_MSG(fmt::format("[WARN][Application::Application()] Loading \"{}\" Error: {}",
            (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string(), Scene::to_string(*err)));
    }
    else
    {
        auto const r_id = std::get<RenderEntityId>(result);
        RenderEntity & r_ent = *_scene->_render_entities.slot(r_id);
        r_ent.transform = glm::mat4x3(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                              glm::vec3(0.0f, 0.0f, 0.0f)) *
                          10.0f;
        DEBUG_MSG(fmt::format("[INFO][Application::Application()] Loading \"{}\" Success",
            (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string()));
    }
    auto scene_commands = _scene->record_gpu_manifest_update();

    _asset_manager = std::make_unique<AssetProcessor>(_gpu_context->device);
    auto const load_result = _asset_manager->load_all(*_scene);
    if (load_result != AssetProcessor::AssetLoadResultCode::SUCCESS)
    {
        DEBUG_MSG(fmt::format("[INFO]Application::Application()] Loading Scene Assets \"{}\" Error: {}",
            (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string(), AssetProcessor::to_string(load_result)));
    }
    else
    {
        DEBUG_MSG(fmt::format("[INFO]Application::Application()] Loading Scene Assets \"{}\" Success",
            (DEFAULT_HARDCODED_PATH / DEFAULT_HARDCODED_FILE).string()));
    }
    auto exc_cmd_list = _asset_manager->record_gpu_load_processing_commands();
    auto cmd_lists = std::array{std::move(scene_commands), std::move(exc_cmd_list)};
    _gpu_context->device.submit_commands({.command_lists = cmd_lists});
    _gpu_context->device.wait_idle();

    _ui_engine = std::make_unique<UIEngine>(*_window, *_asset_manager, _gpu_context.get());

    _renderer = std::make_unique<Renderer>(
        _window.get(), _gpu_context.get(), _scene.get(), _asset_manager.get(), &_ui_engine->imgui_renderer);

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
        update();
        _renderer->render_frame(this->camera_controller.cam_info, this->observer_camera_controller.cam_info, delta_time);
    }
    return 0;
}

void Application::update()
{
    if (_window->size.x == 0 || _window->size.y == 0) { return; }
    _ui_engine->main_update(_gpu_context->settings, *_scene);
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
        DEBUG_MSG(fmt::format("switched enable_observer from {} to {}", _renderer->context->settings.enable_observer,
            !(_renderer->context->settings.enable_observer)));
        _renderer->context->settings.enable_observer = !_renderer->context->settings.enable_observer;
    }
    if (_window->key_just_pressed(GLFW_KEY_J))
    {
        DEBUG_MSG(fmt::format("switched control_observer from {} to {}", control_observer, !(control_observer)));
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
        DEBUG_MSG(fmt::format("switched enable_mesh_shader from {} to {}", _renderer->context->settings.enable_mesh_shader,
            !(_renderer->context->settings.enable_mesh_shader)));
        _renderer->context->settings.enable_mesh_shader = !_renderer->context->settings.enable_mesh_shader;
    }
#endif
    if (_window->key_just_pressed(GLFW_KEY_O))
    {
        DEBUG_MSG(fmt::format("switched observer_show_pass from {} to {}", _renderer->context->settings.observer_show_pass,
            ((_renderer->context->settings.observer_show_pass + 1) % 3)));
        _renderer->context->settings.observer_show_pass = (_renderer->context->settings.observer_show_pass + 1) % 3;
    }
}

Application::~Application()
{
}