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
#include "pgi/pgi.hpp"

struct Renderer
{
    Renderer(Window *window, GPUContext *gpu_context, Scene *scene, AssetProcessor *asset_manager, daxa::ImGuiRenderer *imgui_renderer, UIEngine * ui_engine);
    ~Renderer();

    void compile_pipelines();
    void recreate_framebuffer();
    void clear_select_buffers();
    void window_resized();
    auto create_main_task_graph() -> daxa::TaskGraph;
    auto create_debug_task_graph() -> daxa::TaskGraph;
    auto create_sky_lut_task_graph() -> daxa::TaskGraph;
    void recreate_sky_luts();
    // Return value determines if the frame should be executed.
    auto prepare_frame(
        u32 frame_index,
        CameraInfo const &camera_info, 
        CameraInfo const &observer_camera_info, 
        f32 const delta_time,
        u64 const total_elapsed_us) -> bool;
    void readback_statistics(daxa::TaskGraph & tg);

    daxa::ImageId stbn2d = {};
    daxa::ImageId stbnCosDir = {};

    daxa::ExternalTaskBuffer meshlet_instances = {};
    daxa::ExternalTaskBuffer visible_meshlet_instances = {};
    daxa::ExternalTaskBuffer visible_mesh_instances = {};
    daxa::ExternalTaskBuffer exposure_state = {};
    daxa::BufferId general_readback_buffer = {};

    std::vector<daxa::ExternalTaskBuffer> buffers = {};
    // Images:
    daxa::ExternalTaskImage transmittance = {};
    daxa::ExternalTaskImage multiscattering = {};
    daxa::ExternalTaskImage sky_ibl_cube = {};

    // Render Targets:
    daxa::ExternalTaskImage swapchain_image = {};
    daxa::ExternalTaskImage depth_history = {};
    daxa::ExternalTaskImage path_trace_history = {};
    daxa::ExternalTaskImage normal_history = {};
    daxa::ExternalTaskImage rtao_history = {};
    daxa::ExternalTaskImage rtgi_depth_history = {};
    daxa::ExternalTaskImage rtgi_face_normal_history = {};
    daxa::ExternalTaskImage rtgi_samplecnt_history = {};
    daxa::ExternalTaskImage rtgi_diffuse_history = {};
    daxa::ExternalTaskImage rtgi_diffuse2_history = {};
    daxa::ExternalTaskImage rtgi_statistics_history = {};

    daxa::ExternalTaskImage rtgi_full_samplecount_history = {};
    daxa::ExternalTaskImage rtgi_full_face_normal_history = {};
    daxa::ExternalTaskImage rtgi_full_color_history = {};
    daxa::ExternalTaskImage rtgi_full_statistics_history = {};

    std::vector<daxa::ExternalTaskImage> images = {};
    std::vector<std::pair<daxa::ImageInfo, daxa::ExternalTaskImage>> frame_buffer_images = {};

    VSMState vsm_state = {};
    PGIState pgi_state = {};

    std::unique_ptr<RenderContext> render_context = {};
    Window *window = {};
    GPUContext *gpu_context = {};
    Scene *scene = {};
    UIEngine *ui_engine = {};
    AssetProcessor *asset_manager = {};
    daxa::TaskGraph main_task_graph;
    daxa::TaskGraph debug_task_graph;
    daxa::TaskGraph sky_task_graph;
    daxa::CommandSubmitInfo submit_info = {};
    daxa::ImGuiRenderer* imgui_renderer;
};