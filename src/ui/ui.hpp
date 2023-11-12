#pragma once

#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>

#include "../window.hpp"
#include "../scene/scene.hpp"
#include "../timberdoodle.hpp"

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

        UIEngine(Window &window);
        void main_update(Settings &settings, Scene const & scene);

    private:
        void draw_scenegraph(Scene const & scene);
};