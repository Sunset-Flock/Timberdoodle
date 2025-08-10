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

template<typename T, u32 MAX_SIZE>
struct InlineArray
{
    std::array<T, MAX_SIZE> values = {};
    u32 count = {};
    auto add(T const& v) -> u32
    {
        values[count] = v;
        return v++;
    }
};


enum RasterPipelineIds
{
    RPID_NONE = 0,
    RPID_DRAW_VISBUFFER_0,
    RPID_DRAW_VISBUFFER_1,
    RPID_DRAW_VISBUFFER_2,
    RPID_DRAW_VISBUFFER_3,
    RPID_DRAW_VISBUFFER_4,
    RPID_DRAW_VISBUFFER_5,
    RPID_DRAW_VISBUFFER_6,
    RPID_DRAW_VISBUFFER_7,
    RPID_CULL_DRAW_PAGES_0,
    RPID_CULL_DRAW_PAGES_1,
    RPID_DRAW_SHADER_DEBUG_LINES,
    RPID_DRAW_SHADER_DEBUG_CIRCLES,
    RPID_DRAW_SHADER_DEBUG_RECTANGLES,
    RPID_DRAW_SHADER_DEBUG_AABBS,
    RPID_DRAW_SHADER_DEBUG_BOXES,
    RPID_PGI_DRAW_DEBUG_PROBES,
    RPID_COUNT,
};
struct PipelineSet
{
    std::array<daxa::RasterPipelineCompileInfo2, RPID_COUNT> raster_compile_infos = {};
    std::array<std::shared_ptr<daxa::RasterPipeline>, RPID_COUNT> raster_pipelines = {};
};

// Renderer struct.
// This should idealy handle all rendering related information and functionality.
struct Renderer
{
    Renderer(Window *window, GPUContext *gpu_context, Scene *scene, AssetProcessor *asset_manager, daxa::ImGuiRenderer *imgui_renderer, UIEngine * ui_engine);
    ~Renderer();

    void compile_pipelines();
    void recreate_framebuffer();
    void clear_select_buffers();
    void window_resized();
    auto create_main_task_graph() -> daxa::TaskGraph;
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

    daxa::TaskBuffer meshlet_instances = {};
    daxa::TaskBuffer visible_meshlet_instances = {};
    daxa::TaskBuffer visible_mesh_instances = {};
    daxa::TaskBuffer exposure_state = {};
    daxa::BufferId general_readback_buffer = {};

    std::vector<daxa::TaskBuffer> buffers = {};
    // Images:
    daxa::TaskImage transmittance = {};
    daxa::TaskImage multiscattering = {};
    daxa::TaskImage sky_ibl_cube = {};

    // Render Targets:
    daxa::TaskImage swapchain_image = {};
    daxa::TaskImage depth_history = {};
    daxa::TaskImage path_trace_history = {};
    daxa::TaskImage normal_history = {};
    daxa::TaskImage rtao_history = {};
    daxa::TaskImage rtgi_diffuse_history = {};
    daxa::TaskImage rtgi_depth_history = {};
    daxa::TaskImage rtgi_face_normal_history = {};
    daxa::TaskImage rtgi_samplecnt_history = {};

    std::vector<daxa::TaskImage> images = {};
    std::vector<std::pair<daxa::ImageInfo, daxa::TaskImage>> frame_buffer_images = {};

    VSMState vsm_state = {};
    PGIState pgi_state = {};

    std::unique_ptr<RenderContext> render_context = {};
    Window *window = {};
    GPUContext *gpu_context = {};
    Scene *scene = {};
    UIEngine *ui_engine = {};
    AssetProcessor *asset_manager = {};
    daxa::TaskGraph main_task_graph;
    daxa::TaskGraph sky_task_graph;
    daxa::CommandSubmitInfo submit_info = {};
    daxa::ImGuiRenderer* imgui_renderer;
};