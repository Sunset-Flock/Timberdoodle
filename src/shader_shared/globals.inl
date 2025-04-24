#pragma once

#include "daxa/daxa.inl"

#include "shared.inl"
#include "debug.inl"
#include "readback.inl"
#include "volumetric.inl"
#include "cull_util.inl"
#include "pgi.inl"
#include "scene.inl"
#include "lights.inl"

struct GPUScene
{
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(GPUMeshLodGroup) mesh_lod_groups;
    daxa_BufferPtr(GPUMeshGroup) mesh_groups;
    daxa_BufferPtr(daxa_u32) entity_to_meshgroup;
    daxa_BufferPtr(GPUMaterial) materials;
    daxa_BufferPtr(daxa_f32mat4x3) entity_transforms;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
    daxa_BufferPtr(GPUPointLight) point_lights;
};

struct RenderGlobalData
{
    daxa_u32 hovered_entity_index;
    daxa_u32 selected_entity_index;
    
    // please pass these fields to push constants directly
    // indirection via globals is slow
    GPUScene scene; 
    CameraInfo main_camera;
    CameraInfo main_camera_prev_frame;
    CameraInfo view_camera;
    CameraInfo view_camera_prev_frame;
    daxa_u32 frame_index;
    daxa_u32 frames_in_flight;
    daxa_f32 delta_time;
    daxa_f32vec2 mainview_depth_hiz_physical_size;      
    daxa_f32vec2 mainview_depth_hiz_size;
    Settings settings;
    CullData cull_data;
    SkySettings sky_settings;
    LightSettings light_settings;
    VSMSettings vsm_settings;
    VolumetricSettings volumetric_settings;
    PostprocessSettings postprocess_settings;
    PGISettings pgi_settings;
    GlobalSamplers samplers;
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug;
    daxa_RWBufferPtr(ReadbackValues) readback;
};
DAXA_DECL_BUFFER_PTR(RenderGlobalData)