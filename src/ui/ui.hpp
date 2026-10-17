#pragma once

#include <daxa/gpu_resources.hpp>
#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>
#include <implot.h>

#include "../window.hpp"
#include "../scene/scene.hpp"
#include "../rendering/scene_renderer_context.hpp"
#include "../timberdoodle.hpp"
#include "../scene/asset_processor.hpp"
#include "../application_state.hpp"
#include "ui_shared.hpp"

#include "widgets/scene_graph.hpp"
#include "widgets/property_viewer.hpp"
#include "widgets/camera_path_editor.hpp"
using namespace tido::types;
using namespace tido::ui;

#define IMGUI_UINT_CHECKBOX(VALUE) \
{\
    bool bvalue = VALUE != 0;\
    ImGui::Checkbox(#VALUE, &bvalue);\
    VALUE = bvalue ? 1 : 0;\
}

#define IMGUI_UINT_CHECKBOX2(NAME, VALUE) \
{\
    bool bvalue = VALUE != 0;\
    ImGui::Checkbox(NAME, &bvalue);\
    VALUE = bvalue ? 1 : 0;\
}

struct RenderTimesHistory
{
    template<typename T, i32 size>
    struct ScrollingBuffer
    {
        bool wrapped = {};
        i32 offset = {};
        std::array<T, size> data = {};
        ScrollingBuffer() = default;
        void add_point(T point)
        {
            data[offset] = point;
            wrapped |= (offset + 1) == size;
            offset = (offset + 1) % size;
        }
        void erase()
        {
            data.fill(T());
            wrapped = false;
            offset = 0;
        }
        auto back() 
        {
            i32 back_index = s_cast<i32>(glm::mod(s_cast<f32>(offset - 1), s_cast<f32>(size)));
            return data[back_index]; 
        }
        auto front() 
        { 
            if(wrapped)
            {
                return data[(offset)]; 
            } else {
                return data[0];
            }
        }
    };
    std::array<ScrollingBuffer<ImVec2, 10000>, RenderTimes::GROUP_COUNT> scrolling_ewa = {};
    std::array<ScrollingBuffer<ImVec2, 10000>, RenderTimes::GROUP_COUNT> scrolling_mean = {};
    std::array<ScrollingBuffer<ImVec2, 10000>, RenderTimes::GROUP_COUNT> scrolling_raw = {};
    // Smoothed standard deviation per group (us), sampled with the group series above (same indices).
    std::array<ScrollingBuffer<ImVec2, 10000>, RenderTimes::GROUP_COUNT> scrolling_std_dev = {};
    // Per pass (flat timing index): smoothed average and smoothed standard deviation (us), shorter window.
    std::array<ScrollingBuffer<ImVec2, 2000>, RenderTimes::FLAT_TIMINGS_COUNT> pass_average = {};
    std::array<ScrollingBuffer<ImVec2, 2000>, RenderTimes::FLAT_TIMINGS_COUNT> pass_std_dev = {};
    i32 mean_sample_count = {};
};

// RT Lab convergence measurement analytics UI state (ui.cpp rt_lab_measure_analytics).
struct RtLabMeasureUi
{
    i32 metric = 0; // RT_LAB_MEASURE_METRIC_*
    f32 threshold_percent = 2.0f;
    i32 slice_frame = 0; // 0 = last shown frame
    i32 view_max_frames = 0; // viewer only shows the first N frames, 0 = all
    std::array<char, 128> save_name = {};
    bool save_failed = false;
    i32 delete_index = -1; // result the delete popup asks about
    bool delete_requested = false;
    bool delete_failed = false;
    i32 rename_index = -1; // result the rename popup edits
    bool rename_requested = false;
    std::array<char, 128> rename_name = {};
    bool rename_failed = false;
    // Heatmap options.
    std::array<f32, 2> overlay_above_percent = {10.0f, 0.0f}; // "> X % of tiles" lines, 0 = off
    i32 heatmap_color_mode = 0; // 0 = log10(1 + tiles), 1 = % of tiles per frame
    i32 heatmap_colormap = ImPlotColormap_Viridis;
    bool show_thresholds = false;
    bool show_overlay_lines = true;
    bool fit_y_to_data = true;
    f32 heatmap_height = 400.0f;
    // Heatmap cache, rebuilt when the measurement, the metric or the color mode changes.
    u32 heatmap_generation = 0;
    i32 heatmap_metric = -1;
    i32 heatmap_built_color_mode = -1;
    i32 heatmap_frames = -1;
    f32 heatmap_max = 0.0f;
    i32 heatmap_lowest_row = 0;  // value rows (0 = lowest values) holding any tile
    i32 heatmap_highest_row = 0;
    std::vector<f32> heatmap = {};
};

struct UIEngine
{
    public:
        struct VSMWindows
        {
            bool view_meta_memory = false;
            bool view_reconstructed_shadow_map = false;
            bool view_page_table = false;
        } vsm_windows;
        bool convert_vdb_window = false;
        bool renderer_settings = true;
        bool widget_rt_lab = true;
        bool widget_settings = false;
        bool widget_renderer_statistics = false;
        bool widget_camera_path_editor = false;
        bool widget_scene_interface = true;
        bool widget_property_viewer = true;
        bool demo_window = false;
        bool tg_debug_ui = false;
        u32 magnify_pixels = 7;
        daxa::ImGuiRenderer imgui_renderer = {};
        SceneGraph scene_graph = {};
        PropertyViewer property_viewer = {};
        CameraPathEditor path_editor = {};
        RenderTimesHistory render_times_history = {};
        Window *window = {};
        SceneInterfaceState scene_interface = {};
        daxa::TaskGraphDebugUi main_task_graph_debug_ui = {};

        i32 debug_visualization_index = {};
        i32 debug_visualization_index_override = {};
        bool lights_mark_volume = false;
        i32 pgi_debug_visualization = {};
        i32 rtgi_debug_visualization = {};
        i32 rt_lab_debug_visualization = {};
        bool rt_lab_histogram_log_scale = false;
        RtLabMeasureUi rt_lab_measure_ui = {};
        i32 lights_debug_visualization = {};
        i32 visbuffer_debug_visualization = {};
        i32 vsm_debug_visualization = {};

        UIEngine(Window &window, AssetProcessor & asset_processor, GPUContext * gpu_context);
        ~UIEngine();
        void main_update(RenderContext & render_context, Scene & scene, ApplicationState & app_state, ThreadPool & threadpool);

    private:
        struct DebugCloneUiState
        {
            i32 selected_mip = {};
            i32 selected_array_layer = {};
            i32 selected_sampler = {};
        };
        std::unordered_map<std::string, DebugCloneUiState> debug_clone_states = {};

        static constexpr std::array<std::string_view, s_cast<u32>(ICONS::SIZE)> ICON_TO_PATH
        {
            "deps\\timberdoodle_assets\\ui\\icons\\chevron_up.png",
            "deps\\timberdoodle_assets\\ui\\icons\\chevron_down.png",
            "deps\\timberdoodle_assets\\ui\\icons\\chevron_right.png",
            "deps\\timberdoodle_assets\\ui\\icons\\mesh.png",
            "deps\\timberdoodle_assets\\ui\\icons\\meshgroup.png",
            "deps\\timberdoodle_assets\\ui\\icons\\plus.png",
            "deps\\timberdoodle_assets\\ui\\icons\\minus.png",
            "deps\\timberdoodle_assets\\ui\\icons\\camera.png",
            "deps\\timberdoodle_assets\\ui\\icons\\light.png",
            "deps\\timberdoodle_assets\\ui\\icons\\material.png",
            "deps\\timberdoodle_assets\\ui\\icons\\collection.png",
            "deps\\timberdoodle_assets\\ui\\icons\\sun.png",
        };
        GPUContext * gpu_context;
        bool gather_perm_measurements = true;
        bool show_entire_interval = false;
        f32 text_font_size = 15.0f;
        int selected = {};

        std::vector<daxa::ImageId> icons = {};
        void ui_scene_graph(Scene const & scene);
        void ui_renderer_settings(RenderContext & render_context, ApplicationState & app_state);
        void ui_rt_lab(RenderContext & render_context, ApplicationState & app_state);
        void ui_vsm_textures(RenderContext & render_context);
        void ui_convert_vdb_windw(ApplicationState & app_state, ThreadPool * _threadpool);

        void ui_render_statistics(RenderContext & render_context, ApplicationState & app_state);
        void ui_visbuffer_pipeline_statistics(RenderContext & render_context);
        void ui_pgi_statistics(RenderContext & render_context);
};