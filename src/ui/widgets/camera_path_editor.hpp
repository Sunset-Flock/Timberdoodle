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
#include "../../camera.hpp"

namespace tido
{
    namespace ui
    {
        struct CameraPathEditor
        {
            CameraPathEditor() = default;
            CameraPathEditor(daxa::ImGuiRenderer * renderer);
            void render(RenderContext & render_context, CinematicCamera & camera, CameraController & main_camera);

            std::vector<CameraAnimationKeyframe> current_animation;
            i32 selected_index = -1;
            f32 current_transition_time = 1.0f;
            bool debug_draw_path = false;

            daxa::ImGuiRenderer * renderer = {};
        };
    } // namespace ui
} // namespace tido