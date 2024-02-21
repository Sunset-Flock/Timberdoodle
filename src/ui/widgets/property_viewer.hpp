#pragma once

#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>
#include <array>
#include "../ui_shared.hpp"
#include "../../timberdoodle.hpp"
#include "../../scene/scene.hpp"

namespace tido
{
    namespace ui
    {
        struct RenderInfo
        {
            SkySettings * sky_settings;
            PostprocessSettings * post_settings;
        };
        struct PropertyViewer
        {
          public:
            PropertyViewer() = default;
            PropertyViewer(daxa::ImGuiRenderer * renderer, std::vector<daxa::ImageId> const * icons, daxa::SamplerId linear_sampler);
            void render(RenderInfo const & info);

          private:
            i32 selected = {};
            daxa::ImGuiRenderer * renderer = {};
            daxa::SamplerId linear_sampler = {};
            std::vector<daxa::ImageId> const * icons = {};

            static constexpr std::array selector_icons = {ICONS::SUN, ICONS::CAMERA, ICONS::MESH};
            void draw_sky_settings(SkySettings * sky_settings);
        };
    } // namespace ui
} // namespace tido
