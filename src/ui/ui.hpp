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
        u32 magnify_pixels = 7;
        bool shader_debug_menu = false;
        f32 debug_f32vec4_drag_speed = 0.05f;
        daxa::ImGuiRenderer imgui_renderer = {};
        SceneGraph scene_graph = {};
        PropertyViewer property_viewer = {};

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
        f32 text_font_size = 15.0f;
        int selected = {};

        std::vector<daxa::ImageId> icons = {};
        void ui_scenegraph(Scene const & scene);
        void ui_renderer_settings(Scene const & scene, Settings & settings);
};