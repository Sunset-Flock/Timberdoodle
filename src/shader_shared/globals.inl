#pragma once

#include <daxa/daxa.inl>

#include "shared.inl"
#include "debug.inl"

struct ShaderGlobals
{
    CameraInfo camera;
    CameraInfo observer_camera;
    daxa_u32 frame_index;
    daxa_f32 delta_time;
    Settings settings;
    GlobalSamplers samplers;
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug_draw_info;
};
DAXA_DECL_BUFFER_PTR(ShaderGlobals)