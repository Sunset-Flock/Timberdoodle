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
#include "ao.inl"
#include "rtgi.inl"

struct GPUScene
{
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(GPUMeshLodGroup) mesh_lod_groups;
    daxa_BufferPtr(GPUMeshGroup) mesh_groups;
    daxa_BufferPtr(daxa_u32) entity_to_meshgroup;
    daxa_BufferPtr(GPUMaterial) materials;
    daxa_u32 material_count;
    daxa_BufferPtr(daxa_f32mat4x3) entity_transforms;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
    daxa_BufferPtr(GPUPointLight) point_lights;
    daxa_BufferPtr(GPUSpotLight) spot_lights;
};

#define MARK_SELECTED_MODE_ENTITY 0
#define MARK_SELECTED_MODE_MESH 1
#define MARK_SELECTED_MODE_MESHLET 2
#define MARK_SELECTED_MODE_TRIANGLE 3

struct RenderGlobalData
{
    // UI Written Data:
    daxa_u32 hovered_entity_index;
    daxa_u32 selected_entity_index;
    daxa_u32 selected_mesh_index;
    daxa_u32 selected_meshlet_in_mesh_index;
    daxa_u32 selected_triangle_in_meshlet_index;
    daxa_i32 selected_mark_mode;
    daxa_f32vec2 cursor_uv;

    // Global Textures
    daxa_ImageViewId stbn2d;
    daxa_ImageViewId stbnCosDir;
    
    // Renderer Written Data:
    GPUScene scene; // Passing scene directly into push can yield good perf gains, avoid this field.
    CameraInfo main_camera;
    CameraInfo main_camera_prev_frame;
    CameraInfo view_camera;
    CameraInfo view_camera_prev_frame;
    daxa_u64 total_elapsed_us;
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
    AoSettings ao_settings;
    RtgiSettings rtgi_settings;
    GlobalSamplers samplers;
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug;
    daxa_RWBufferPtr(ReadbackValues) readback;
};
DAXA_DECL_BUFFER_PTR(RenderGlobalData)