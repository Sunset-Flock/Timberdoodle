#pragma once

#include "daxa/daxa.inl"

#include "shared.inl"
#include "debug.inl"
#include "readback.inl"
#include "volumetric.inl"
#include "cull_util.inl"
#include "ddgi.inl"

struct RenderGlobalData
{
    CameraInfo camera;
    CameraInfo observer_camera;
    CameraInfo camera_prev_frame;
    CameraInfo observer_camera_prev_frame;
    daxa_u32 frame_index;
    daxa_u32 frames_in_flight;
    daxa_f32 delta_time;
    daxa_f32vec2 mainview_depth_hiz_physical_size;      
    daxa_f32vec2 mainview_depth_hiz_size;
    Settings settings;
    CullData cull_data;
    SkySettings sky_settings;
    VSMSettings vsm_settings;
    VolumetricSettings volumetric_settings;
    PostprocessSettings postprocess_settings;
    DDGISettings ddgi_settings;
    daxa_BufferPtr(SkySettings) sky_settings_ptr;
    GlobalSamplers samplers;
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug;
    daxa_RWBufferPtr(ReadbackValues) readback;
};
DAXA_DECL_BUFFER_PTR(RenderGlobalData)