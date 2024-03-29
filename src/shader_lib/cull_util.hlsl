#pragma once

#include "daxa/daxa.inl"

#include "cull_util.glsl"

func is_meshlet_visible_vsm(
    CameraInfo camera,
    MeshletInstance meshlet_inst,
    daxa_f32mat4x3* entity_combined_transforms,
    GPUMesh* meshes,
    Texture2DArray<uint4> hiz,
    uint cascade) -> bool
{
    Ptr<GPUMesh> mesh = meshes + meshlet_inst.entity_index;
    if (meshlet_inst.meshlet_index >= mesh->meshlet_count)
    {
        return false;
    }


}