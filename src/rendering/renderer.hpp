#pragma once

#include <string>
#include <daxa/utils/imgui.hpp>

#include "../window.hpp"
#include "../shader_shared/geometry.inl"
#include "../shader_shared/readback.inl"
#include "../scene/scene.hpp"
#include "../scene/asset_processor.hpp"
#include "../ui/ui.hpp"

#include "../gpu_context.hpp"
#include "../camera.hpp"
#include "scene_renderer_context.hpp"
#include "virtual_shadow_maps/vsm_state.hpp"


// Renderer struct.
// This should idealy handle all rendering related information and functionality.
struct Renderer
{
    Renderer(Window *window, GPUContext *context, Scene *scene, AssetProcessor *asset_manager, daxa::ImGuiRenderer *imgui_renderer, UIEngine * ui_engine);
    ~Renderer();

    void compile_pipelines();
    void recreate_framebuffer();
    void clear_select_buffers();
    void window_resized();
    auto create_main_task_graph() -> daxa::TaskGraph;
    auto create_sky_lut_task_graph() -> daxa::TaskGraph;
    void recreate_sky_luts();
    void render_frame(
        CameraInfo const &camera_info, 
        CameraInfo const &observer_camera_info, 
        f32 const delta_time);
    void readback_statistics(daxa::TaskGraph & tg);

    daxa::TaskBuffer zero_buffer = {};

    daxa::TaskBuffer shader_debug_buffer = {};
    daxa::TaskBuffer meshlet_instances = {};
    daxa::TaskBuffer meshlet_instances_last_frame = {};
    daxa::TaskBuffer visible_meshlet_instances = {};
    daxa::TaskBuffer visible_mesh_instances = {};
    daxa::TaskBuffer luminance_average = {};
    daxa::TaskBuffer general_readback_buffer = {};

    std::vector<daxa::TaskBuffer> buffers = {};
    // Images:
    daxa::TaskImage transmittance = {};
    daxa::TaskImage multiscattering = {};
    daxa::TaskImage sky_ibl_cube = {};

    // Render Targets:
    daxa::TaskImage swapchain_image = {};

    std::vector<daxa::TaskImage> images = {};
    std::vector<std::pair<daxa::ImageInfo, daxa::TaskImage>> frame_buffer_images = {};

    VSMState vsm_state = {};

    std::unique_ptr<RenderContext> render_context = {};
    Window *window = {};
    GPUContext *context = {};
    Scene *scene = {};
    UIEngine *ui_engine = {};
    AssetProcessor *asset_manager = {};
    daxa::TaskGraph main_task_graph;
    daxa::TaskGraph sky_task_graph;
    daxa::CommandSubmitInfo submit_info = {};
    daxa::ImGuiRenderer* imgui_renderer;
};