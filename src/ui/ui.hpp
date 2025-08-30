#pragma once

#include <daxa/gpu_resources.hpp>
#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>

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
    i32 mean_sample_count = {};
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
        bool renderer_settings = true;
        bool widget_settings = false;
        bool widget_renderer_statistics = false;
        bool widget_camera_path_editor = false;
        bool widget_scene_interface = true;
        bool widget_property_viewer = true;
        bool demo_window = false;
        bool tg_debug_ui = false;
        u32 magnify_pixels = 7;
        bool shader_debug_menu = false;
        f32 fixed_camera_x_rotation_speed = 0.0f;
        f32 debug_f32vec4_drag_speed = 0.05f;
        daxa::ImGuiRenderer imgui_renderer = {};
        SceneGraph scene_graph = {};
        PropertyViewer property_viewer = {};
        CameraPathEditor path_editor = {};
        RenderTimesHistory render_times_history = {};
        Window *window = {};
        SceneInterfaceState scene_interface = {};

        i32 debug_visualization_index = {};
        i32 debug_visualization_index_override = {};
        bool lights_mark_volume = false;
        i32 pgi_debug_visualization = {};
        i32 rtgi_debug_visualization = {};
        i32 rtao_debug_visualization = {};
        i32 lights_debug_visualization = {};
        i32 visbuffer_debug_visualization = {};
        i32 vsm_debug_visualization = {};

        UIEngine(Window &window, AssetProcessor & asset_processor, GPUContext * gpu_context);
        ~UIEngine();
        void main_update(GPUContext const & gpu_context, RenderContext & render_context, Scene & scene, ApplicationState & app_state, Window & window);

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
        void ui_renderer_settings(Scene const & scene, RenderContext & render_context, ApplicationState & app_state);
        void ui_vsm_textures(RenderContext & render_context, ApplicationState & app_state);

        void ui_render_statistics(Scene const & scene, RenderContext & render_context, ApplicationState & app_state);
        void ui_visbuffer_pipeline_statistics(Scene const & scene, RenderContext & render_context, ApplicationState & app_state);
        void ui_pgi_statistics(Scene const & scene, RenderContext & render_context, ApplicationState & app_state);
};