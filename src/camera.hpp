#pragma once

#include "timberdoodle.hpp"
using namespace tido::types;

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "window.hpp"
#include "shader_shared/shared.inl"

void hermite(
    f32vec3& pos,
    f32vec3& vel, 
    f32 x, 
    f32vec3 p0,
    f32vec3 p1, 
    f32vec3 v0,
    f32vec3 v1);

void catmull_rom(
    f32vec3& pos,
    f32vec3& vel,
    f32 x,
    f32vec3 p0,
    f32vec3 p1, 
    f32vec3 p2,
    f32vec3 p3);

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

struct CameraAnimationKeyframe
{
    glm::fquat rotation;
    f32vec3 position;
    f32 transition_time;
};

struct CinematicCamera
{
    CinematicCamera() = default;
    void update_keyframes(std::vector<CameraAnimationKeyframe> && keyframes);
    void process_input(Window &window, f32 dt);
    void set_keyframe(i32 keyframe, f32 keyframe_progress);
    auto make_camera_info(Settings const & settings) const -> CameraInfo;

    f32vec3 up = {0.f, 0.f, 1.0f};
    glm::fquat forward = {};
    f32vec3 position = {};
    f32 fov = 70.0f;
    f32 near = 0.01f;
    bool override_keyframe = {};

    f32 current_keyframe_time = 0.0f;
    u32 current_keyframe_index = 0;
    std::vector<CameraAnimationKeyframe> path_keyframes = {};
};