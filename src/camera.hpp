#pragma once

#include "timberdoodle.hpp"
using namespace tido::types;

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "window.hpp"
#include "shader_shared/shared.inl"

struct CameraController
{
    void process_input(Window &window, f32 dt);
    auto make_camera_info(Settings const & settings) const -> CameraInfo;

    bool bZoom = false;
    f32 fov = 70.0f;
    f32 near = 0.1f;
    f32 cameraSwaySpeed = 0.05f;
    f32 translationSpeed = 10.0f;
    f32vec3 up = {0.f, 0.f, 1.0f};
    f32vec3 forward = {0.962, -0.25, -0.087};
    f32vec3 position = {-22.f, 4.f, 6.f};
    f32 yaw = 0.0f;
    f32 pitch = 0.0f;
};

struct AnimationKeyframe
{
    glm::fquat start_rotation;
    glm::fquat end_rotation;

    f32vec3 start_position;
    f32vec3 first_control_point;
    f32vec3 second_control_point;
    f32vec3 end_position;
    f32 transition_time;
};

struct CinematicCamera
{
    CinematicCamera() = default;
    void update_keyframes(std::vector<AnimationKeyframe> && keyframes);
    void process_input(Window &window, f32 dt);
    void set_keyframe(i32 keyframe, f32 keyframe_progress);
    auto make_camera_info(Settings const & settings) const -> CameraInfo;

    f32vec3 up = {0.f, 0.f, 1.0f};
    glm::fquat forward = {};
    f32vec3 position = {};
    f32 fov = 70.0f;
    f32 near = 0.1f;
    bool override_keyframe = {};

    f32 current_keyframe_time = 0.0f;
    u32 current_keyframe_index = 0;
    std::vector<AnimationKeyframe> path_keyframes = {};
};