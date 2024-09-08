#pragma once

#include "daxa/daxa.inl"
#include "../shader_shared/shared.inl"
#include "../shader_shared/cull_util.inl"
#include "../shader_lib/debug.glsl"
#include "../shader_shared/globals.inl"
#include "../shader_shared/geometry.inl"
#include "../shader_shared/geometry_pipeline.inl"
#include "../shader_lib/vsm_util.glsl"
#include "../shader_shared/vsm_shared.inl"

// bool is_tri_out_of_frustum(CameraInfo camera, daxa_f32vec3 tri[3])
// {
//     const daxa_f32vec3 frustum_planes[5] = {
//         camera.right_plane_normal,
//         camera.left_plane_normal,
//         camera.top_plane_normal,
//         camera.bottom_plane_normal,
//         camera.near_plane_normal,
//     };
//     bool out_of_frustum = false;
//     for (uint i = 0; i < 5; ++i)
//     {
//         bool tri_out_of_plane = true;
//         for (uint ti = 0; ti < 3; ++ti)
//         {
//             tri_out_of_plane = tri_out_of_plane && dot((tri[ti] - camera.position), frustum_planes[i]) > 0.0f;
//         }
//         out_of_frustum = out_of_frustum || tri_out_of_plane;
//     }
//     return out_of_frustum;
// }

bool is_meshlet_drawn_in_first_pass(
    MeshletInstance meshlet_inst,
    daxa_BufferPtr(daxa_u32) first_pass_meshlets_bitfield_offsets,
    U32ArenaBufferRef first_pass_meshlets_bitfield_arena
)
{
    const uint first_pass_meshgroup_bitfield_offset = deref_i(first_pass_meshlets_bitfield_offsets, meshlet_inst.entity_index);
    if ((first_pass_meshgroup_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID) && 
        (first_pass_meshgroup_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED))
    {
        const uint mesh_instance_bitfield_offset_offset = first_pass_meshgroup_bitfield_offset + meshlet_inst.in_mesh_group_index;
        // Offset is valid, need to check if mesh instance offset is valid now.
        const uint first_pass_mesh_instance_bitfield_offset = first_pass_meshlets_bitfield_arena.uints[mesh_instance_bitfield_offset_offset];
        if ((first_pass_mesh_instance_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID) && 
            (first_pass_mesh_instance_bitfield_offset != FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED))
        {
            // Offset is valid, must check bitfield now.
            uint in_bitfield_u32_index = meshlet_inst.meshlet_index / 32 + first_pass_mesh_instance_bitfield_offset;
            const uint in_u32_bit = meshlet_inst.meshlet_index % 32;
            const uint in_u32_mask = 1u << in_u32_bit;
            const uint bitfield_u32 = first_pass_meshlets_bitfield_arena.uints[in_bitfield_u32_index];
            const bool meshlet_drawn_first_pass = (bitfield_u32 & in_u32_mask) != 0;
            // DEBUG_INDEX(
            //     mesh_instance_bitfield_offset_offset,
            //     0, 
            //     first_pass_meshlets_bitfield_arena.offsets_section_size - 1,
            //     "ASSERT ERROR: OUT OF BOUNDS ACCESS IN is_meshlet_drawn_in_first_pass on");
            if (meshlet_drawn_first_pass)
            {
                return true;
            }
        }
    }
    return false;
}

BoundingSphere calculate_meshlet_ws_bounding_sphere(
    daxa_f32mat4x4 model_matrix,
    BoundingSphere model_space_bounding_sphere
)
{
    const float model_scaling_x_squared = dot(model_matrix[0],model_matrix[0]);
    const float model_scaling_y_squared = dot(model_matrix[1],model_matrix[1]);
    const float model_scaling_z_squared = dot(model_matrix[2],model_matrix[2]);
    const float radius_scaling = sqrt(max(max(model_scaling_x_squared,model_scaling_y_squared), model_scaling_z_squared));
    BoundingSphere ret;
    ret.radius = radius_scaling * model_space_bounding_sphere.radius;
    ret.center = mul(model_matrix, daxa_f32vec4(model_space_bounding_sphere.center, 1)).xyz;
    return ret;
}

bool is_ws_sphere_out_of_frustum(CameraInfo camera, BoundingSphere ws_bounding_sphere)
{
    const daxa_f32vec3 frustum_planes[5] = {
        camera.right_plane_normal,
        camera.left_plane_normal,
        camera.top_plane_normal,
        camera.bottom_plane_normal,
        camera.near_plane_normal,
    };
    const bool cull_near = camera.is_orthogonal;
    const float ortho_plane_offset_ws = camera.orthogonal_half_ws_width;
    bool out_of_frustum = false;
    for (uint i = 0; i < (cull_near ? 5 : 4); ++i)
    {
        out_of_frustum = out_of_frustum || (dot((ws_bounding_sphere.center - camera.position), frustum_planes[i]) - ws_bounding_sphere.radius - ortho_plane_offset_ws) > 0.0f;
    }
    return out_of_frustum;
}

bool is_sphere_out_of_frustum(CameraInfo camera, daxa_f32mat4x4 model_matrix, BoundingSphere ms_bounding_sphere)
{
    BoundingSphere ws_bs = calculate_meshlet_ws_bounding_sphere(model_matrix, ms_bounding_sphere);
    return is_ws_sphere_out_of_frustum(camera, ws_bs);
}

struct NdcAABB
{
    daxa_f32vec3 ndc_min;
    daxa_f32vec3 ndc_max;
};

NdcAABB calculate_ndc_aabb(
    CameraInfo camera,
    daxa_f32mat4x4 model_matrix,
    AABB aabb
)
{
    bool initialized_min_max = false;
    NdcAABB ret;
    for (int z = -1; z <= 1; z += 2)
    {
        for (int y = -1; y <= 1; y += 2)
        {
            for (int x = -1; x <= 1; x += 2)
            {
                const daxa_f32vec3 model_corner_position = aabb.center + aabb.size * daxa_f32vec3(x,y,z) * 0.5f;
                const daxa_f32vec4 worldspace_corner_position = mul(model_matrix, daxa_f32vec4(model_corner_position,1));
                const daxa_f32vec4 clipspace_corner_position = mul(camera.view_proj, worldspace_corner_position);
                const daxa_f32vec3 ndc_corner_position = clipspace_corner_position.xyz / clipspace_corner_position.w;
                ret.ndc_min.x = !initialized_min_max ? ndc_corner_position.x : min(ndc_corner_position.x, ret.ndc_min.x);
                ret.ndc_min.y = !initialized_min_max ? ndc_corner_position.y : min(ndc_corner_position.y, ret.ndc_min.y);
                ret.ndc_min.z = !initialized_min_max ? ndc_corner_position.z : min(ndc_corner_position.z, ret.ndc_min.z);
                ret.ndc_max.x = !initialized_min_max ? ndc_corner_position.x : max(ndc_corner_position.x, ret.ndc_max.x);
                ret.ndc_max.y = !initialized_min_max ? ndc_corner_position.y : max(ndc_corner_position.y, ret.ndc_max.y);
                ret.ndc_max.z = !initialized_min_max ? ndc_corner_position.z : max(ndc_corner_position.z, ret.ndc_max.z);
                initialized_min_max = true;
            }
        }
    }

    ret.ndc_min.x = max(ret.ndc_min.x, -1.0f);
    ret.ndc_min.y = max(ret.ndc_min.y, -1.0f);
    ret.ndc_max.x = min(ret.ndc_max.x,  1.0f);
    ret.ndc_max.y = min(ret.ndc_max.y,  1.0f);
    return ret;
}

bool is_ndc_aabb_hiz_depth_occluded(
    CameraInfo camera,
    NdcAABB meshlet_ndc_aabb,
    daxa_u32vec2 hiz_res,
    daxa_ImageViewId hiz
)
{
    const daxa_f32vec2 f_hiz_resolution = hiz_res;
    const daxa_f32vec2 min_uv = (meshlet_ndc_aabb.ndc_min.xy + 1.0f) * 0.5f;
    const daxa_f32vec2 max_uv = (meshlet_ndc_aabb.ndc_max.xy + 1.0f) * 0.5f;
    const daxa_f32vec2 min_texel_i = floor(clamp(f_hiz_resolution * min_uv, daxa_f32vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const daxa_f32vec2 max_texel_i = floor(clamp(f_hiz_resolution * max_uv, daxa_f32vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const float pixel_range = max(max_texel_i.x - min_texel_i.x + 1.0f, max_texel_i.y - min_texel_i.y + 1.0f);
    const float mip = ceil(log2(max(2.0f, pixel_range))) - 1 /* we want one mip lower, as we sample a quad */;

    // The calculation above gives us a mip level, in which the a 2x2 quad in that mip is just large enough to fit the ndc bounds.
    // When the ndc bounds are shofted from the alignment of that mip levels grid however, we need an even larger quad.
    // We check if the quad at its current position within that mip level fits that quad and if not we move up one mip.
    // This will give us the tightest fit.
    int imip = int(mip);
    const daxa_i32vec2 min_corner_texel = daxa_i32vec2(min_texel_i) >> imip;
    const daxa_i32vec2 max_corner_texel = daxa_i32vec2(max_texel_i) >> imip;
    if (any(greaterThan(max_corner_texel - min_corner_texel, daxa_i32vec2(1)))) {
        imip += 1;
    }
    const daxa_i32vec2 quad_corner_texel = daxa_i32vec2(min_texel_i) >> imip;
    const daxa_i32vec2 texel_bounds = max(daxa_i32vec2(0,0), (daxa_i32vec2(f_hiz_resolution) >> imip) - 1);

    const daxa_f32vec4 fetch = daxa_f32vec4(
        __texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(0,0), daxa_i32vec2(0,0), texel_bounds), imip).x,
        __texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(0,1), daxa_i32vec2(0,0), texel_bounds), imip).x,
        __texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(1,0), daxa_i32vec2(0,0), texel_bounds), imip).x,
        __texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(1,1), daxa_i32vec2(0,0), texel_bounds), imip).x
    );
    const float conservative_depth = min(min(fetch.x,fetch.y), min(fetch.z, fetch.w));
    const bool depth_cull = meshlet_ndc_aabb.ndc_max.z < conservative_depth;
    return depth_cull;
}

bool is_ndc_aabb_hiz_opacity_occluded(
    CameraInfo camera,
    NdcAABB meshlet_ndc_aabb,
    daxa_ImageViewId hiz,
    daxa_u32 array_layer
)
{
    const daxa_f32vec2 f_hiz_resolution = daxa_f32vec2(camera.screen_size >> 1 /*hiz is half res*/);
    const daxa_f32vec2 min_uv = (meshlet_ndc_aabb.ndc_min.xy + 1.0f) * 0.5f;
    const daxa_f32vec2 max_uv = (meshlet_ndc_aabb.ndc_max.xy + 1.0f) * 0.5f;
    const daxa_f32vec2 min_texel_i = floor(clamp(f_hiz_resolution * min_uv, daxa_f32vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const daxa_f32vec2 max_texel_i = floor(clamp(f_hiz_resolution * max_uv, daxa_f32vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const float pixel_range = max(max_texel_i.x - min_texel_i.x + 1.0f, max_texel_i.y - min_texel_i.y + 1.0f);
    const float mip = ceil(log2(max(2.0f, pixel_range))) - 1 /* we want one mip lower, as we sample a quad */;

    // The calculation above gives us a mip level, in which the a 2x2 quad in that mip is just large enough to fit the ndc bounds.
    // When the ndc bounds are shofted from the alignment of that mip levels grid however, we need an even larger quad.
    // We check if the quad at its current position within that mip level fits that quad and if not we move up one mip.
    // This will give us the tightest fit.
    int imip = int(mip);
    const daxa_i32vec2 min_corner_texel = daxa_i32vec2(min_texel_i) >> imip;
    const daxa_i32vec2 max_corner_texel = daxa_i32vec2(max_texel_i) >> imip;
    if (any(greaterThan(max_corner_texel - min_corner_texel, daxa_i32vec2(1)))) {
        imip += 1;
    }
    const daxa_i32vec2 quad_corner_texel = daxa_i32vec2(min_texel_i) >> imip;
    const daxa_i32vec2 texel_bounds = max(daxa_i32vec2(0,0), (daxa_i32vec2(f_hiz_resolution) >> imip) - 1);

    Texture2DArray<uint> thiz = Texture2DArray<uint>::get(hiz);

    const daxa_u32vec4 fetch = daxa_u32vec4(
        thiz.Load(daxa_u32vec4(clamp(quad_corner_texel + daxa_i32vec2(0,0), daxa_i32vec2(0,0), texel_bounds), array_layer, imip)).x,
        thiz.Load(daxa_u32vec4(clamp(quad_corner_texel + daxa_i32vec2(0,1), daxa_i32vec2(0,0), texel_bounds), array_layer, imip)).x,
        thiz.Load(daxa_u32vec4(clamp(quad_corner_texel + daxa_i32vec2(1,0), daxa_i32vec2(0,0), texel_bounds), array_layer, imip)).x,
        thiz.Load(daxa_u32vec4(clamp(quad_corner_texel + daxa_i32vec2(1,1), daxa_i32vec2(0,0), texel_bounds), array_layer, imip)).x
    );
    const bool no_valid_pages_in_ndc = (fetch.x | fetch.y | fetch.z | fetch.w) == 0;
    return no_valid_pages_in_ndc;
}

bool is_meshlet_occluded(
    ShaderDebugBufferHead* debug,
    CameraInfo camera,
    MeshletInstance meshlet_inst,
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_u32vec2 hiz_res,
    daxa_ImageViewId hiz
)
{
    GPUMesh mesh_data = deref_i(meshes, meshlet_inst.mesh_index);
    if (mesh_data.mesh_buffer.value == 0)
    {
        return true;
    }

    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(deref_i(entity_combined_transforms, meshlet_inst.entity_index));
    BoundingSphere model_bounding_sphere = deref_i(mesh_data.meshlet_bounds, meshlet_inst.meshlet_index);

    BoundingSphere ws_bs = calculate_meshlet_ws_bounding_sphere(model_matrix, model_bounding_sphere);
    if (is_ws_sphere_out_of_frustum(camera, ws_bs))
    {
        if (debug && false)
        {
            ShaderDebugCircleDraw circle;
            circle.position = ws_bs.center;
            circle.radius = ws_bs.radius;
            circle.color = daxa_f32vec3(1,1,0);
            circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            debug_draw_circle(debug, circle);
        }
        return true;
    }

    AABB meshlet_aabb = deref_i(mesh_data.meshlet_aabbs, meshlet_inst.meshlet_index);
    NdcAABB meshlet_ndc_aabb = calculate_ndc_aabb(camera, model_matrix, meshlet_aabb);
    const bool depth_cull = is_ndc_aabb_hiz_depth_occluded(camera, meshlet_ndc_aabb, hiz_res, hiz);

    #if (defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus))
    if (depth_cull)
    {
        ShaderDebugAABBDraw aabb1;
        aabb1.position = mul(model_matrix, daxa_f32vec4(meshlet_aabb.center,1)).xyz;
        aabb1.size = mul(model_matrix, daxa_f32vec4(meshlet_aabb.size,0)).xyz;
        aabb1.color = daxa_f32vec3(0.1, 0.5, 1);
        aabb1.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_aabb(GLOBALS.debug, aabb1);
        {
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (ndc_max - ndc_min);
            rectangle.center = ndc_min + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(0, 1, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
        {
            const daxa_f32vec2 min_r = quad_corner_texel << imip;
            const daxa_f32vec2 max_r = (quad_corner_texel + 2) << imip;
            const daxa_f32vec2 min_r_uv = min_r / f_hiz_resolution;
            const daxa_f32vec2 max_r_uv = max_r / f_hiz_resolution;
            const daxa_f32vec2 min_r_ndc = min_r_uv * 2.0f - 1.0f;
            const daxa_f32vec2 max_r_ndc = max_r_uv * 2.0f - 1.0f;
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (daxa_f32vec3(max_r_ndc, ndc_max.z) - daxa_f32vec3(min_r_ndc, ndc_min.z));
            rectangle.center = daxa_f32vec3(min_r_ndc, ndc_min.z) + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(1, 0, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
    }
    #endif

    return depth_cull;
}

bool is_mesh_occluded(
    ShaderDebugBufferHead* debug,
    CameraInfo camera,
    MeshInstance mesh_inst,
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_u32vec2 hiz_res,
    daxa_ImageViewId hiz
)
{
    GPUMesh mesh_data = deref_i(meshes, mesh_inst.mesh_index);
    if (mesh_data.mesh_buffer.value == 0)
    {
        return true;
    }

    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(deref_i(entity_combined_transforms, mesh_inst.entity_index));
    // BoundingSphere model_bounding_sphere = deref_i(mesh_data.meshlet_bounds, mesh_inst.meshlet_index);
    // BoundingSphere ws_bs = calculate_meshlet_ws_bounding_sphere(model_matrix, model_bounding_sphere);
    // if (is_ws_sphere_out_of_frustum(camera, ws_bs))
    // {
    //     if (debug && false)
    //     {
    //         ShaderDebugCircleDraw circle;
    //         circle.position = ws_bs.center;
    //         circle.radius = ws_bs.radius;
    //         circle.color = daxa_f32vec3(1,1,0);
    //         circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
    //         debug_draw_circle(debug, circle);
    //     }
    //     return true;
    // }

    AABB meshlet_aabb = mesh_data.aabb;
    NdcAABB meshlet_ndc_aabb = calculate_ndc_aabb(camera, model_matrix, meshlet_aabb);
    const bool depth_cull = is_ndc_aabb_hiz_depth_occluded(camera, meshlet_ndc_aabb, hiz_res, hiz);

    if (false && depth_cull && debug)
    {
        ShaderDebugAABBDraw aabb1;
        aabb1.position = mul(model_matrix, daxa_f32vec4(meshlet_aabb.center,1)).xyz;
        aabb1.size = mul(model_matrix, daxa_f32vec4(meshlet_aabb.size,0)).xyz;
        aabb1.color = daxa_f32vec3(0.1, 0.5, 1);
        aabb1.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_aabb(debug, aabb1);
        {
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (meshlet_ndc_aabb.ndc_max - meshlet_ndc_aabb.ndc_min);
            rectangle.center = meshlet_ndc_aabb.ndc_min + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(0, 1, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(debug, rectangle);
        }
        // {
        //     const daxa_f32vec2 min_r = quad_corner_texel << imip;
        //     const daxa_f32vec2 max_r = (quad_corner_texel + 2) << imip;
        //     const daxa_f32vec2 min_r_uv = min_r / f_hiz_resolution;
        //     const daxa_f32vec2 max_r_uv = max_r / f_hiz_resolution;
        //     const daxa_f32vec2 min_r_ndc = min_r_uv * 2.0f - 1.0f;
        //     const daxa_f32vec2 max_r_ndc = max_r_uv * 2.0f - 1.0f;
        //     ShaderDebugRectangleDraw rectangle;
        //     const daxa_f32vec3 rec_size = (daxa_f32vec3(max_r_ndc, ndc_max.z) - daxa_f32vec3(min_r_ndc, ndc_min.z));
        //     rectangle.center = daxa_f32vec3(min_r_ndc, ndc_min.z) + (rec_size * 0.5);
        //     rectangle.span = rec_size.xy;
        //     rectangle.color = daxa_f32vec3(1, 0, 1);
        //     rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
        //     debug_draw_rectangle(debug, rectangle);
        // }
    }

    return depth_cull;
}

bool is_mesh_occluded_vsm(
    CameraInfo camera,
    MeshInstance mesh_inst,
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_ImageViewId hiz,
    daxa_u32 cascade
)
{
    GPUMesh mesh_data = deref_i(meshes, mesh_inst.mesh_index);
    if (mesh_data.mesh_buffer.value == 0)
    {
        return true;
    }
    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(deref_i(entity_combined_transforms, mesh_inst.entity_index));
    // BoundingSphere model_bounding_sphere = mesh_data.bounding_sphere;
    // if (is_sphere_out_of_frustum(camera, model_matrix, model_bounding_sphere))
    // {
    //     #if defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus)
    //         ShaderDebugCircleDraw circle;
    //         circle.position = ws_center;
    //         circle.radius = scaled_radius;
    //         circle.color = daxa_f32vec3(1,1,0);
    //         circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
    //         debug_draw_circle(GLOBALS.debug, circle);
    //     #endif
    //     // return true;
    // }

    AABB mesh_aabb = mesh_data.aabb;
    NdcAABB mesh_ndc_aabb = calculate_ndc_aabb(camera, model_matrix, mesh_aabb);
    const bool page_opacity_cull = is_ndc_aabb_hiz_opacity_occluded(camera, mesh_ndc_aabb, hiz, cascade);

    #if (defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus))
    if (page_opacity_cull)
    {
        ShaderDebugAABBDraw aabb1;
        aabb1.position = mul(model_matrix, daxa_f32vec4(meshlet_aabb.center,1)).xyz;
        aabb1.size = mul(model_matrix, daxa_f32vec4(meshlet_aabb.size,0)).xyz;
        aabb1.color = daxa_f32vec3(0.1, 0.5, 1);
        aabb1.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_aabb(GLOBALS.debug, aabb1);
        {
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (ndc_max - ndc_min);
            rectangle.center = ndc_min + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(0, 1, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
        {
            const daxa_f32vec2 min_r = quad_corner_texel << imip;
            const daxa_f32vec2 max_r = (quad_corner_texel + 2) << imip;
            const daxa_f32vec2 min_r_uv = min_r / f_hiz_resolution;
            const daxa_f32vec2 max_r_uv = max_r / f_hiz_resolution;
            const daxa_f32vec2 min_r_ndc = min_r_uv * 2.0f - 1.0f;
            const daxa_f32vec2 max_r_ndc = max_r_uv * 2.0f - 1.0f;
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (daxa_f32vec3(max_r_ndc, ndc_max.z) - daxa_f32vec3(min_r_ndc, ndc_min.z));
            rectangle.center = daxa_f32vec3(min_r_ndc, ndc_min.z) + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(1, 0, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
    }
    #endif

    return page_opacity_cull;
}

func is_triangle_invisible_micro_triangle(float2 ndc_min, float2 ndc_max, float2 resolution) -> bool
{
    // Just to be save :)
    let delta = 1.0 / 256.0f;
    let sample_grid_min = (ndc_min * 0.5f + 0.5f) * resolution - 0.5f - delta;
    let sample_grid_max = (ndc_max * 0.5f + 0.5f) * resolution - 0.5f + delta;
    // Checks if the min and the max positions are right next to the same sample grid line.
    // If we are next to the same sample grid line in one dimension we are not rasterized.
    let prim_visible = !any(equal(floor(sample_grid_max), floor(sample_grid_min)));
    return !prim_visible && all(greaterThan(ndc_min, -float2(0.99999,0.99999))) && all(lessThan(ndc_max, float2(0.99999,0.99999)));
}

func is_triangle_backfacing(float4 tri_vert_clip_positions[3]) -> bool
{
    let is_backface =
        determinant(float3x3(
            tri_vert_clip_positions[0].xyw,
            tri_vert_clip_positions[1].xyw,
            tri_vert_clip_positions[2].xyw)) >= 0;
    return is_backface;
}

bool is_triangle_hiz_occluded(
    CameraInfo camera,
    float3 ndc_positions[3],
    daxa_u32vec2 hiz_res,
    daxa_ImageViewId hiz
)
{
    NdcAABB aabb;
    aabb.ndc_max = max(ndc_positions[0].xyz, max(ndc_positions[1].xyz, ndc_positions[2].xyz));
    aabb.ndc_min = min(ndc_positions[0].xyz, min(ndc_positions[1].xyz, ndc_positions[2].xyz));
    const bool depth_cull = is_ndc_aabb_hiz_depth_occluded(camera, aabb, hiz_res, hiz);
    return depth_cull;
}

bool is_meshlet_occluded_vsm(
    CameraInfo camera,
    MeshletInstance meshlet_inst,
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_ImageViewId hiz,
    daxa_u32 cascade
)
{
    GPUMesh mesh_data = deref_i(meshes, meshlet_inst.mesh_index);
    if (mesh_data.mesh_buffer.value == 0)
    {
        return true;
    }
    if (meshlet_inst.meshlet_index >= mesh_data.meshlet_count)
    {
        return true;
    }

    daxa_f32mat4x4 model_matrix = mat_4x3_to_4x4(deref_i(entity_combined_transforms, meshlet_inst.entity_index));
    BoundingSphere model_bounding_sphere = deref_i(mesh_data.meshlet_bounds, meshlet_inst.meshlet_index);
    if (is_sphere_out_of_frustum(camera, model_matrix, model_bounding_sphere))
    {
        #if defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus)
            ShaderDebugCircleDraw circle;
            circle.position = ws_center;
            circle.radius = scaled_radius;
            circle.color = daxa_f32vec3(1,1,0);
            circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            debug_draw_circle(GLOBALS.debug, circle);
        #endif
        // return true;
    }

    AABB meshlet_aabb = deref_i(mesh_data.meshlet_aabbs, meshlet_inst.meshlet_index);
    NdcAABB meshlet_ndc_aabb = calculate_ndc_aabb(camera, model_matrix, meshlet_aabb);
    const bool page_opacity_cull = is_ndc_aabb_hiz_opacity_occluded(camera, meshlet_ndc_aabb, hiz, cascade);

    #if (defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus))
    if (page_opacity_cull)
    {
        ShaderDebugAABBDraw aabb1;
        aabb1.position = mul(model_matrix, daxa_f32vec4(meshlet_aabb.center,1)).xyz;
        aabb1.size = mul(model_matrix, daxa_f32vec4(meshlet_aabb.size,0)).xyz;
        aabb1.color = daxa_f32vec3(0.1, 0.5, 1);
        aabb1.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_aabb(GLOBALS.debug, aabb1);
        {
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (ndc_max - ndc_min);
            rectangle.center = ndc_min + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(0, 1, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
        {
            const daxa_f32vec2 min_r = quad_corner_texel << imip;
            const daxa_f32vec2 max_r = (quad_corner_texel + 2) << imip;
            const daxa_f32vec2 min_r_uv = min_r / f_hiz_resolution;
            const daxa_f32vec2 max_r_uv = max_r / f_hiz_resolution;
            const daxa_f32vec2 min_r_ndc = min_r_uv * 2.0f - 1.0f;
            const daxa_f32vec2 max_r_ndc = max_r_uv * 2.0f - 1.0f;
            ShaderDebugRectangleDraw rectangle;
            const daxa_f32vec3 rec_size = (daxa_f32vec3(max_r_ndc, ndc_max.z) - daxa_f32vec3(min_r_ndc, ndc_min.z));
            rectangle.center = daxa_f32vec3(min_r_ndc, ndc_min.z) + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = daxa_f32vec3(1, 0, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
    }
    #endif

    return page_opacity_cull;
}