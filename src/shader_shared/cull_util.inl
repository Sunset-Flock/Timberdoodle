#pragma once

#include <daxa/daxa.inl>

// TODO(pahrens): add frustum planes for frustum culling.
// TODO(pahrens): put this struct into the globals, and make a variant for the current culling matrix.
struct FrustumInfo
{
    daxa_f32mat4x4 view;
    daxa_f32mat4x4 projection;
    daxa_f32mat4x4 view_projection;
    daxa_f32vec3 camera_position;
    daxa_f32vec4 camera_direction_rotor;
    daxa_f32 near_plane;
    daxa_f32 aspect_ratio;
    daxa_f32 vertical_fov;
    daxa_f32 horizontal_fov;
};

#define CULL_MESHLETS_WORKGROUP_X 128
#define TASK_SHADER_WORKGROUP_X 32