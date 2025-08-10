#pragma once

#include "camera.hpp"
#include "scene/scene.hpp"

struct ApplicationState
{
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
    u32 frame_index = 0;
    f32 delta_time = 0.016666f;
    u64 total_elapsed_us = 0;
    f32 time_taken_cpu_windowing = 0.016666f;
    f32 time_taken_cpu_application = 0.016666f;
    f32 time_taken_cpu_wait_for_gpu = 0.016666f;
    f32 time_taken_cpu_renderer_prepare = 0.016666f;
    f32 time_taken_cpu_renderer_record = 0.016666f;
    std::chrono::time_point<std::chrono::steady_clock> startup_time_point = {};
    std::chrono::time_point<std::chrono::steady_clock> last_time_point = {};
    std::string desired_scene_path = {};
};