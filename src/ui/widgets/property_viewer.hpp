#pragma once

#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>
#include <array>
#include "../ui_shared.hpp"
#include "../../timberdoodle.hpp"
#include "../../scene/scene.hpp"
#include "../../rendering/scene_renderer_context.hpp"

namespace tido
{
    namespace ui
    {
        struct PropertyViewer
        {
            PropertyViewer() = default;
            PropertyViewer(daxa::ImGuiRenderer * renderer, std::vector<daxa::ImageId> const * icons, daxa::SamplerId linear_sampler);
            void render(SceneInterfaceState & scene_interface, Scene & scene, RenderContext & render_context);

            i32 selected = {};
            daxa::ImGuiRenderer * renderer = {};
            daxa::SamplerId linear_sampler = {};
            std::vector<daxa::ImageId> const * icons = {};

            static constexpr std::array selector_icons = {ICONS::SUN, ICONS::CAMERA, ICONS::MESH};
        };
    } // namespace ui
} // namespace tido
