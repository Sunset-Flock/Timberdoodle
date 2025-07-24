#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/shared.inl"
#include "../shader_lib/debug.glsl"
#include "../shader_shared/globals.inl"
#include "../shader_shared/geometry.inl"
#include "../shader_shared/geometry_pipeline.inl"

func select_lod(RenderGlobalData* render_data, GPUMeshLodGroup* mesh_lod_groups, GPUMesh* meshes, uint mesh_lod_group_index, float4x4 combined_transform, daxa::f32 resolution, daxa_f32vec3 camera_pos, daxa::f32 acceptable_pixel_error, daxa::i32 lod_override) -> daxa::u32
{
    /// ===== Select LOD ======
    // To select the lod we use calculate an acceptable error for each mesh transformed in world position.
    // We then calculate the error that each lod would have and select the highest lod that is lower than the acceptable error.
    // Error:
    //   Lod error is a value from 0-1.
    //   Its the mean squared error of mesh surface positions compared to lod0
    //   The unit of the error is model space difference divided by model world space size, so it is irrelevant how large the model is in worldspace.
    // Error Estimation for Lod Selection
    //   We can very coarsely calculate how many pixels would change from a lod change:
    //   - mesh lod aabb pixel size * mesh lod error
    //   We can then determine a pixel error that we do not want to overstep, lets say 2 or 4.
    //   We calculate the pixel error for each lod and pick the highest one with acceptable error.
    // Off screen handling
    //   We still want meshes loded properly if they are off screen
    //   Important for raytracing
    //   Important for shadowmaps
    //   Because of this we can not simply project the aabb into viewspace and calculate the real pixel coverage.
    //   Instead, a simplified method will be used to estimate a rough pixel size each mesh would be, 
    //   based on distance, fov and resolution only, no projection.
    // Mesh Pixel Size Approximation
    //   This should be VERY fast to be able to handle tens of thousands of meshes
    //   It can be very coarse, we can investigate finer lodding later
    /// ===== Select LOD ======

    // Iterate over lods, calculate estimated pixel error, select last lod with acceptable error.
    daxa::u32 selected_lod_lod = 0;
    GPUMeshLodGroup mesh_lod_group = mesh_lod_groups[mesh_lod_group_index];
    let meshes_base_offset = mesh_lod_group_index * MAX_MESHES_PER_LOD_GROUP;

    if (lod_override >= 0)
    {
        selected_lod_lod = min(lod_override, mesh_lod_group.lod_count - 1u);
    }
    else
    {
        // Calculate perspective and size based on lod 0:
        GPUMesh mesh_lod0 = meshes[meshes_base_offset + 0]; 
        daxa::f32 aabb_extent_x = length(combined_transform[0]) * mesh_lod0.aabb.size.x;
        daxa::f32 aabb_extent_y = length(combined_transform[1]) * mesh_lod0.aabb.size.y;
        daxa::f32 aabb_extent_z = length(combined_transform[2]) * mesh_lod0.aabb.size.z;
        daxa::f32 aabb_rough_extent = max(max(aabb_extent_x, aabb_extent_y), aabb_extent_z);

        daxa::f32vec3 aabb_center = mul(combined_transform, daxa::f32vec4(mesh_lod0.aabb.center, 1.0f)).xyz;
        daxa::f32 aabb_rough_camera_distance = max(0.0f, length(aabb_center - camera_pos) - 0.5f * aabb_rough_extent);

        // Assumes a 90 fov camera for simplicity
        daxa::f32 fov90_distance_to_screen_ratio = 2.0f;
        daxa::f32 pixel_size_at_1m = fov90_distance_to_screen_ratio / resolution;
        daxa::f32 aabb_size_at_1m = (aabb_rough_extent / aabb_rough_camera_distance);
        daxa::f32 rough_aabb_pixel_size = aabb_size_at_1m / pixel_size_at_1m;

        for (daxa::u32 lod = 1; lod < mesh_lod_group.lod_count; ++lod)
        {
            GPUMesh mesh = meshes[meshes_base_offset + lod];
            daxa::f32 rough_pixel_error = rough_aabb_pixel_size * mesh.lod_error;
            if (rough_pixel_error < acceptable_pixel_error)
            {
                selected_lod_lod = lod;
            }
            else
            {
                break;
            }
            // gpu_context->shader_debug_context.cpu_debug_aabb_draws.push_back(ShaderDebugAABBDraw{
            //     .position = std::bit_cast<daxa_f32vec3>(aabb_center),
            //     .size = daxa_f32vec3(aabb_extent_x, aabb_extent_y, aabb_extent_z),
            //     .color = daxa_f32vec3(1, 0, 0),
            //     .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
            // });
        }
    }
    return selected_lod_lod;
}