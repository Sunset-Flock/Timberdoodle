#pragma once

#include <string>
#include <daxa/utils/imgui.hpp>

#include "../window.hpp"
#include "../shader_shared/geometry.inl"
#include "../scene/scene.hpp"
#include "../scene/asset_processor.hpp"

#include "../gpu_context.hpp"
#include "scene_renderer_context.hpp"
#include "virtual_shadow_maps/vsm_state.hpp"

struct CameraController
{
    void process_input(Window &window, f32 dt);
    void update_matrices(Settings const & settings);
    auto make_camera_info(Settings const & settings) const -> CameraInfo;

    bool bZoom = false;
    f32 fov = 70.0f;
    f32 near = 0.1f;
    f32 cameraSwaySpeed = 0.05f;
    f32 translationSpeed = 10.0f;
    f32vec3 up = {0.f, 0.f, 1.0f};
    f32vec3 forward = {0.962, -0.25, -0.087};
    f32vec3 position = {-22.f, 4.f, 6.f};
    f32 yaw = 0.0f;
    f32 pitch = 0.0f;
};

// Renderer struct.
// This should idealy handle all rendering related information and functionality.
struct Renderer
{
    Renderer(Window *window, GPUContext *context, Scene *scene, AssetProcessor *asset_manager, daxa::ImGuiRenderer *imgui_renderer);
    ~Renderer();

    void compile_pipelines(bool allow_mesh_shader, bool allow_slang);
    void recreate_framebuffer();
    void clear_select_buffers();
    void window_resized();
    auto create_main_task_graph() -> daxa::TaskGraph;
    auto create_sky_lut_task_graph() -> daxa::TaskGraph;
    void recreate_sky_luts();
    void render_frame(
        CameraController const &camera_info, 
        CameraController const &observer_camera_info, 
        f32 const delta_time,
        SceneDraw scene_draw);

    VSMState vsm_state = {};

    daxa::TaskBuffer zero_buffer = {};

    daxa::TaskBuffer shader_debug_buffer = {};
    daxa::TaskBuffer meshlet_instances = {};
    daxa::TaskBuffer meshlet_instances_last_frame = {};
    daxa::TaskBuffer visible_meshlet_instances = {};
    daxa::TaskBuffer luminance_average = {};

    std::vector<daxa::TaskBuffer> buffers = {};
    // Images:
    daxa::TaskImage transmittance = {};
    daxa::TaskImage multiscattering = {};
    daxa::TaskImage sky_ibl_cube = {};

    // Render Targets:
    daxa::TaskImage swapchain_image = {};

    std::vector<daxa::TaskImage> images = {};
    std::vector<std::pair<daxa::ImageInfo, daxa::TaskImage>> frame_buffer_images = {};

    std::unique_ptr<RenderContext> render_context = {};
    Window *window = {};
    GPUContext *context = {};
    Scene *scene = {};
    AssetProcessor *asset_manager = {};
    daxa::TaskGraph main_task_graph;
    daxa::TaskGraph sky_task_graph;
    daxa::CommandSubmitInfo submit_info = {};
    daxa::ImGuiRenderer* imgui_renderer;
};