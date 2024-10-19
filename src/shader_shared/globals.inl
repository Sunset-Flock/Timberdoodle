#pragma once

#include "daxa/daxa.inl"

#include "shared.inl"
#include "debug.inl"
#include "readback.inl"
#include "volumetric.inl"

struct RenderGlobalData
{
    daxa_f32mat4x3 test[2];
    CameraInfo camera;
    CameraInfo observer_camera;
    daxa_u32 frame_index;
    daxa_f32 delta_time;
    Settings settings;
    SkySettings sky_settings;
    VSMSettings vsm_settings;
    VolumetricSettings volumetric_settings;
    PostprocessSettings postprocess_settings;
    daxa_BufferPtr(SkySettings) sky_settings_ptr;
    GlobalSamplers samplers;
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug;
    daxa_RWBufferPtr(ReadbackValues) readback;
};
DAXA_DECL_BUFFER_PTR(RenderGlobalData)