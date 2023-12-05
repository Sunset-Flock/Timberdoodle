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
using namespace tido::types;
using namespace tido::ui;

struct UIEngine
{

    public:
        bool widget_settings = false;
        bool widget_renderer_statistics = false;
        bool widget_scene_hierarchy = false;
        tido::ui::SceneGraph scene_graph{};
        daxa::ImGuiRenderer imgui_renderer = {};

        UIEngine(Window &window, AssetProcessor & asset_processor, GPUContext * context);
        ~UIEngine();
        void main_update(Settings &settings, Scene const & scene);

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
        };
        GPUContext * context;
        f32 text_font_size = 15.0f;

        std::vector<daxa::ImageId> icons = {};
        void draw_scenegraph(Scene const & scene);
};