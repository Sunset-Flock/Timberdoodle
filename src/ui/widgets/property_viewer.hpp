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
#include "../../application_state.hpp"

namespace tido
{
    namespace ui
    {
        struct PropertyViewer
        {
            PropertyViewer() = default;
            PropertyViewer(daxa::ImGuiRenderer * renderer, std::vector<daxa::ImageId> const * icons, daxa::SamplerId linear_sampler);
            void render(SceneInterfaceState & scene_interface, Scene & scene, RenderContext & render_context, ApplicationState & app_state);

            i32 selected = {};
            daxa::ImGuiRenderer * renderer = {};
            daxa::SamplerId linear_sampler = {};
            std::vector<daxa::ImageId> const * icons = {};
            f32 fixed_camera_x_rotation_speed = 0.0f;

            // Programmed-move button state. Movement is a camera-relative velocity
            // (x = right, y = up, z = forward), applied every frame for
            // auto_move_duration seconds. Optional yaw/pitch rotation (deg/sec) is
            // applied over the same interval. When auto_move_screenshot is set a
            // screenshot fires on the last moving frame; when auto_move_return_to_start
            // is set the camera pose captured at move start is restored one frame later
            // (so the screenshot still captures the end of the move).
            f32vec3 auto_move_velocity = {0.1f, 0.0f, 0.0f};
            f32 auto_move_yaw_speed = 0.0f;
            f32 auto_move_pitch_speed = 0.0f;
            f32 auto_move_duration = 1.0f;
            f32 auto_move_time_remaining = 0.0f;
            bool auto_move_screenshot = true;
            bool auto_move_return_to_start = true;
            f32vec3 auto_move_start_position = {};
            f32 auto_move_start_yaw = 0.0f;
            f32 auto_move_start_pitch = 0.0f;
            bool auto_move_return_pending = false;

            static constexpr std::array selector_icons = {ICONS::SUN, ICONS::CAMERA, ICONS::MESH};
        };
    } // namespace ui
} // namespace tido
