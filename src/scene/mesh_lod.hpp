#pragma once

#include <filesystem>
#include <meshoptimizer.h>
#include <fastgltf/types.hpp>
#include <mutex>

#include "../timberdoodle.hpp"
#include "../gpu_context.hpp"
#include "../shader_shared/geometry.inl"
#include "scene.hpp"

// ===== Timberdoodle Mesh Lod System =====
// 
// - When Tido loads meshes at runtime, it will attempt to generate lods for each loaded mesh.
// - The lods of a mesh form a "MeshLodGroup".
// - To make Indexing simpler, each MeshLodGroup always allocates 16 meshes. 
//   - allows for mesh_lod_group_index = mesh_index / 16
//   - allows for lod_of_mesh = mesh_index % 16
//
// - Tido attempts to generate lods that each have half the triangle count as the previous
//   - While generating tido will check if the resulting mesh of a simplification exceeds an error measure.
//   - When the error measure gets too large, tido stops generating lods for the mesh.
//   - Most Meshes will generate 8-12 lods.
// - Error estimation incudes vertex position error relative to model bounds
//   - Error estimation includes vertex normal error relative to the average vertex distance within the model
//   - The average vertex distance is recalculated for each simplification
//   - Normal weighting MUST be relative to vertex distance for each lod to be consistent
//   - Error approximates visual difference / model volume
//
// - When the scene collects mesh instances to be drawn and iterates over all entities, tido performs lod selection
//   - for lod selection, tido projects the AABB of each mesh onto a image plane that is directed towards the mesh from the main camera position.
//   - view plane is directed towards the mesh so we get camera view independent lod selection.
//     - makes lods consistent, no pop when camera sways
//     - also makes lod selection consistent for things behind the camera
//   - using the projectes view plane size, tido estimates the potential pixel error in main camera resolution
//   - tido selects the highest lod that still has an estimated pixel error below 1.
//     - error threshold can be changed via ui. Larger values can lead to much better perf in some scenes
//     - 1 pixel error as default leads to impercieveable  lod switches.
//
// ===== Timberdoodle Mesh Lod System =====

auto select_lod(RenderGlobalData const& render_data, MeshLodGroupManifestEntry const& mesh_lod_group, usize mesh_lod_group_index, RenderEntity const* r_ent) -> u32
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
    u32 selected_lod_lod = 0;

    if (render_data.settings.lod_override >= 0)
    {
        selected_lod_lod = std::min(static_cast<u32>(render_data.settings.lod_override), mesh_lod_group.runtime->lod_count - 1u);
    }
    else
    {
        // Calculate perspective and size based on lod 0:
        GPUMesh const & mesh_lod0 = mesh_lod_group.runtime->lods[0]; 
        f32 const aabb_extent_x = glm::length(r_ent->combined_transform[0]) * mesh_lod0.aabb.size.x;
        f32 const aabb_extent_y = glm::length(r_ent->combined_transform[1]) * mesh_lod0.aabb.size.y;
        f32 const aabb_extent_z = glm::length(r_ent->combined_transform[2]) * mesh_lod0.aabb.size.z;
        f32 const aabb_rough_extent = std::max(std::max(aabb_extent_x, aabb_extent_y), aabb_extent_z);

        glm::vec3 const aabb_center = r_ent->combined_transform * glm::vec4(std::bit_cast<glm::vec3>(mesh_lod0.aabb.center), 1.0f);
        f32 const aabb_rough_camera_distance = std::max(0.0f, glm::length(aabb_center - std::bit_cast<glm::vec3>(render_data.main_camera.position)) - 0.5f * aabb_rough_extent);

        f32 const rough_resolution = std::max(render_data.settings.render_target_size.x, render_data.settings.render_target_size.y);

        // Assumes a 90 fov camera for simplicity
        f32 const fov90_distance_to_screen_ratio = 2.0f;
        f32 const pixel_size_at_1m = fov90_distance_to_screen_ratio / rough_resolution;
        f32 const aabb_size_at_1m = (aabb_rough_extent / aabb_rough_camera_distance);
        f32 const rough_aabb_pixel_size = aabb_size_at_1m / pixel_size_at_1m;

        for (u32 lod = 1; lod < mesh_lod_group.runtime->lod_count; ++lod)
        {
            GPUMesh const & mesh = mesh_lod_group.runtime->lods[lod]; 
            f32 const rough_pixel_error = rough_aabb_pixel_size * mesh.lod_error;
            if (rough_pixel_error < render_data.settings.lod_acceptable_pixel_error)
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
    u32 mesh_index = mesh_lod_group_index * MAX_MESHES_PER_LOD_GROUP + selected_lod_lod;
    return mesh_index;
}