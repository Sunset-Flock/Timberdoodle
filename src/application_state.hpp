#pragma once

#include "camera.hpp"
#include "scene/scene.hpp"
#include "asteroids/asteroids.hpp"

struct ApplicationState
{
    AsteroidSimulation simulation = {};
    CameraController camera_controller = {};
    CameraController observer_camera_controller = {};
    CinematicCamera cinematic_camera = {};
    RenderEntityId dynamic_ball = {};
    RenderEntityId root_id = {};
    bool draw_observer = false;
    bool control_observer = false;
    bool use_preset_camera = false;
    bool keep_running = true;
    bool reset_observer = false;
    bool decompose_bistro = false;
    f32 delta_time = 0.016666f;
    std::chrono::time_point<std::chrono::steady_clock> last_time_point = {};
};