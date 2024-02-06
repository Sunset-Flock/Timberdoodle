#pragma once

#include <string>
#include <daxa/utils/imgui.hpp>

#include "../window.hpp"
#include "../shader_shared/asset.inl"
#include "../scene/scene.hpp"
#include "../scene/asset_processor.hpp"

#include "../gpu_context.hpp"

// Renderer struct.
// This should idealy handle all rendering related information and functionality.
struct Renderer
{
    Renderer(Window *window, GPUContext *context, Scene *scene, AssetProcessor *asset_manager, daxa::ImGuiRenderer *imgui_renderer);
    ~Renderer();

    void compile_pipelines();
    void recreate_framebuffer();
    void clear_select_buffers();
    void window_resized();
    auto create_main_task_graph() -> daxa::TaskGraph;
    void update_settings();
    void render_frame(CameraInfo const &camera_info, CameraInfo const &observer_camera_info, f32 const delta_time);

    daxa::TaskBuffer zero_buffer = {};

    daxa::TaskBuffer shader_debug_buffer = {};
    daxa::TaskBuffer meshlet_instances = {};
    daxa::TaskBuffer meshlet_instances_last_frame = {};
    daxa::TaskBuffer visible_meshlet_instances = {};

    std::vector<daxa::TaskBuffer> buffers = {};

    // Render Targets:
    daxa::TaskImage swapchain_image = {};
    daxa::TaskImage depth = {};
    daxa::TaskImage visbuffer = {};
    daxa::TaskImage debug_image = {};

    std::vector<daxa::TaskImage> images = {};
    std::vector<std::pair<daxa::ImageInfo, daxa::TaskImage>> frame_buffer_images = {};

    Window *window = {};
    GPUContext *context = {};
    Scene *scene = {};
    AssetProcessor *asset_manager = {};
    daxa::TaskGraph main_task_graph;
    daxa::CommandSubmitInfo submit_info = {};
    daxa::ImGuiRenderer* imgui_renderer;
};