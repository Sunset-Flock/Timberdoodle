#pragma once

#include <daxa/gpu_resources.hpp>
#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>

#include "../window.hpp"
#include "../scene/scene.hpp"
#include "../timberdoodle.hpp"
#include "../scene/asset_processor.hpp"
#include "ui_shared.hpp"

#include "widgets/scene_graph.hpp"
#include "widgets/property_viewer.hpp"
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

struct PerfMeasurements
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
    std::array<ScrollingBuffer<ImVec2, 10000>, 3> scrolling_ewa = {};
    std::array<ScrollingBuffer<ImVec2, 10000>, 3> scrolling_mean = {};
    std::array<ScrollingBuffer<ImVec2, 10000>, 3> scrolling_raw = {};
    std::array<f32, 11> vsm_timings_ewa = {};
    std::array<f32, 11> vsm_timings_mean = {};
    std::array<f32, 11> vsm_timings_raw = {};
    i32 mean_sample_count = {};
};

struct UIEngine
{
    public:
        bool renderer_settings = true;
        bool widget_settings = false;
        bool widget_renderer_statistics = false;
        bool widget_scene_hierarchy = true;
        bool widget_property_viewer = true;
        bool demo_window = false;
        bool vsm_debug_menu = false;
        bool aurora_debug_menu = true;
        u32 magnify_pixels = 7;
        u32 perf_sample_count = 0;
        bool shader_debug_menu = false;
        f32 debug_f32vec4_drag_speed = 0.05f;
        daxa::ImGuiRenderer imgui_renderer = {};
        SceneGraph scene_graph = {};
        PropertyViewer property_viewer = {};
        PerfMeasurements measurements = {};

        UIEngine(Window &window, AssetProcessor & asset_processor, GPUContext * context);
        ~UIEngine();
        void main_update(RenderContext & render_ctx, Scene const & scene);

    private:
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
        GPUContext * context;
        bool gather_perm_measurements = true;
        bool show_entire_interval = false;
        bool continuously_regenerage_aurora = false;
        bool temporal_aurora_accumulation = true;
        f32 text_font_size = 15.0f;
        int selected = {};

        std::vector<daxa::ImageId> icons = {};
        void ui_scenegraph(Scene const & scene);
        void ui_renderer_settings(Scene const & scene, Settings & settings);
};