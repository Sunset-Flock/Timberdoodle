#pragma once

#include "camera.hpp"
#include "scene/scene.hpp"

struct ApplicationState
{
    CameraController camera_controller = {};
    CameraController observer_camera_controller = {};
    CinematicCamera cinematic_camera = {};
    RenderEntityId dynamic_ball = {};
    bool draw_observer = false;
    bool control_observer = false;
    bool use_preset_camera = false;
    bool keep_running = true;
    bool reset_observer = false;
    f32 delta_time = 0.016666f;
    std::chrono::time_point<std::chrono::steady_clock> last_time_point = {};
};