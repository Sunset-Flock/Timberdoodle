#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"
#include "../shader_shared/globals.inl"
#include "../shader_shared/asset.inl"

bool is_out_of_frustum(vec3 ws_center, float ws_radius)
{
    const vec3 frustum_planes[5] = {
        deref(push.uses.globals).camera.right_plane_normal,
        deref(push.uses.globals).camera.left_plane_normal,
        deref(push.uses.globals).camera.top_plane_normal,
        deref(push.uses.globals).camera.bottom_plane_normal,
        deref(push.uses.globals).camera.near_plane_normal,
    };
    bool out_of_frustum = false;
    for (uint i = 0; i < 5; ++i)
    {
        out_of_frustum = out_of_frustum || (dot((ws_center - deref(push.uses.globals).camera.pos), frustum_planes[i]) - ws_radius) > 0.0f;
    }
    return out_of_frustum;
}

bool is_tri_out_of_frustum(vec3 tri[3])
{
    const vec3 frustum_planes[5] = {
        deref(push.uses.globals).camera.right_plane_normal,
        deref(push.uses.globals).camera.left_plane_normal,
        deref(push.uses.globals).camera.top_plane_normal,
        deref(push.uses.globals).camera.bottom_plane_normal,
        deref(push.uses.globals).camera.near_plane_normal,
    };
    bool out_of_frustum = false;
    for (uint i = 0; i < 5; ++i)
    {
        bool tri_out_of_plane = true;
        for (uint ti = 0; ti < 3; ++ti)
        {
            tri_out_of_plane = tri_out_of_plane && dot((tri[ti] - deref(push.uses.globals).camera.pos), frustum_planes[i]) > 0.0f;
        }
        out_of_frustum = out_of_frustum || tri_out_of_plane;
    }
    return out_of_frustum;
}

bool is_meshlet_occluded(
    MeshletInstance meshlet_inst,
    EntityMeshletVisibilityBitfieldOffsetsView entity_meshlet_visibility_bitfield_offsets,
    daxa_BufferPtr(daxa_u32) entity_meshlet_visibility_bitfield_arena,
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms,
    daxa_BufferPtr(GPUMesh) meshes,
    daxa_ImageViewId hiz
)
{
    GPUMesh mesh_data = deref(meshes[meshlet_inst.mesh_index]);
    if (meshlet_inst.meshlet_index >= mesh_data.meshlet_count)
    {
        return true;
    }
    const uint bitfield_uint_offset = meshlet_inst.meshlet_index / 32;
    const uint bitfield_uint_bit = 1u << (meshlet_inst.meshlet_index % 32);
    const uint entity_arena_offset = entity_meshlet_visibility_bitfield_offsets.entity_offsets[meshlet_inst.entity_index].mesh_bitfield_offset[meshlet_inst.in_meshgroup_index];
    if (entity_arena_offset != ENT_MESHLET_VIS_OFFSET_UNALLOCATED && entity_arena_offset != ENT_MESHLET_VIS_OFFSET_EMPTY)
    {
        const uint mask = deref(entity_meshlet_visibility_bitfield_arena[entity_arena_offset + bitfield_uint_offset]);
        const bool visible_last_frame = (mask & bitfield_uint_bit) != 0;
        if (visible_last_frame)
        {
            return true;
        }
    }
    // daxa_f32vec3 center;
    // daxa_f32 radius;
    mat4x4 model_matrix = mat_4x3_to_4x4(deref(entity_combined_transforms[meshlet_inst.entity_index]));
    mat4x4 view_proj = deref(push.uses.globals).camera.view_proj;
    const float model_scaling_x_squared = dot(model_matrix[0],model_matrix[0]);
    const float model_scaling_y_squared = dot(model_matrix[1],model_matrix[1]);
    const float model_scaling_z_squared = dot(model_matrix[2],model_matrix[2]);
    const float radius_scaling = sqrt(max(max(model_scaling_x_squared,model_scaling_y_squared), model_scaling_z_squared));
    BoundingSphere bounds = deref(mesh_data.meshlet_bounds[meshlet_inst.meshlet_index]);
    AABB meshlet_aabb = deref(mesh_data.meshlet_aabbs[meshlet_inst.meshlet_index]);
    const float scaled_radius = radius_scaling * bounds.radius;
    const vec3 ws_center = (model_matrix * vec4(bounds.center, 1)).xyz;

    if (is_out_of_frustum(ws_center, scaled_radius))
    {
        #if defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus)
            ShaderDebugCircleDraw circle;
            circle.position = ws_center;
            circle.radius = scaled_radius;
            circle.color = vec3(1,1,0);
            circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            debug_draw_circle(GLOBALS.shader_debug_context, circle);
        #endif
        return true;
    }

    bool initialized_min_max = false;
    vec3 ndc_min;
    vec3 ndc_max;
    for (int z = -1; z <= 1; z += 2)
    {
        for (int y = -1; y <= 1; y += 2)
        {
            for (int x = -1; x <= 1; x += 2)
            {
                const vec3 model_corner_position = meshlet_aabb.center + meshlet_aabb.size * vec3(x,y,z) * 0.5f;
                const vec4 worldspace_corner_position = model_matrix * vec4(model_corner_position,1);
                const vec4 clipspace_corner_position = view_proj * worldspace_corner_position;
                const vec3 ndc_corner_position = clipspace_corner_position.xyz / clipspace_corner_position.w;
                ndc_min.x = !initialized_min_max ? ndc_corner_position.x : min(ndc_corner_position.x, ndc_min.x);
                ndc_min.y = !initialized_min_max ? ndc_corner_position.y : min(ndc_corner_position.y, ndc_min.y);
                ndc_min.z = !initialized_min_max ? ndc_corner_position.z : min(ndc_corner_position.z, ndc_min.z);
                ndc_max.x = !initialized_min_max ? ndc_corner_position.x : max(ndc_corner_position.x, ndc_max.x);
                ndc_max.y = !initialized_min_max ? ndc_corner_position.y : max(ndc_corner_position.y, ndc_max.y);
                ndc_max.z = !initialized_min_max ? ndc_corner_position.z : max(ndc_corner_position.z, ndc_max.z);
                initialized_min_max = true;
            }
        }
    }

    ndc_min.x = max(ndc_min.x, -1.0f);
    ndc_min.y = max(ndc_min.y, -1.0f);
    ndc_max.x = min(ndc_max.x,  1.0f);
    ndc_max.y = min(ndc_max.y,  1.0f);

    const vec2 f_hiz_resolution = vec2(deref(push.uses.globals).settings.render_target_size >> 1 /*hiz is half res*/);
    const vec2 min_uv = (ndc_min.xy + 1.0f) * 0.5f;
    const vec2 max_uv = (ndc_max.xy + 1.0f) * 0.5f;
    const vec2 min_texel_i = floor(clamp(f_hiz_resolution * min_uv, vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const vec2 max_texel_i = floor(clamp(f_hiz_resolution * max_uv, vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const float pixel_range = max(max_texel_i.x - min_texel_i.x + 1.0f, max_texel_i.y - min_texel_i.y + 1.0f);
    const float mip = ceil(log2(max(2.0f, pixel_range))) - 1 /* we want one mip lower, as we sample a quad */;

    // The calculation above gives us a mip level, in which the a 2x2 quad in that mip is just large enough to fit the ndc bounds.
    // When the ndc bounds are shofted from the alignment of that mip levels grid however, we need an even larger quad.
    // We check if the quad at its current position within that mip level fits that quad and if not we move up one mip.
    // This will give us the tightest fit.
    int imip = int(mip);
    const ivec2 min_corner_texel = ivec2(min_texel_i) >> imip;
    const ivec2 max_corner_texel = ivec2(max_texel_i) >> imip;
    if (any(greaterThan(max_corner_texel - min_corner_texel, ivec2(1)))) {
        imip += 1;
    }
    const ivec2 quad_corner_texel = ivec2(min_texel_i) >> imip;
    const ivec2 texel_bounds = max(ivec2(0,0), (ivec2(f_hiz_resolution) >> imip) - 1);

    const vec4 fetch = vec4(
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(0,0), ivec2(0,0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(0,1), ivec2(0,0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(1,0), ivec2(0,0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(1,1), ivec2(0,0), texel_bounds), imip).x
    );
    const float conservative_depth = min(min(fetch.x,fetch.y), min(fetch.z, fetch.w));
    const bool depth_cull = ndc_max.z < conservative_depth;

    #if defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus)
    if (depth_cull)
    {
        ShaderDebugAABBDraw aabb1;
        aabb1.position = (model_matrix * vec4(meshlet_aabb.center,1)).xyz;
        aabb1.size = (model_matrix * vec4(meshlet_aabb.size,0)).xyz;
        aabb1.color = vec3(0.1, 0.5, 1);
        aabb1.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_aabb(GLOBALS.shader_debug_context, aabb1);
        {
            ShaderDebugRectangleDraw rectangle;
            const vec3 rec_size = (ndc_max - ndc_min);
            rectangle.center = ndc_min + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = vec3(0, 1, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(deref(push.uses.globals).shader_debug_context, rectangle);
        }
        {
            const vec2 min_r = quad_corner_texel << imip;
            const vec2 max_r = (quad_corner_texel + 2) << imip;
            const vec2 min_r_uv = min_r / f_hiz_resolution;
            const vec2 max_r_uv = max_r / f_hiz_resolution;
            const vec2 min_r_ndc = min_r_uv * 2.0f - 1.0f;
            const vec2 max_r_ndc = max_r_uv * 2.0f - 1.0f;
            ShaderDebugRectangleDraw rectangle;
            const vec3 rec_size = (vec3(max_r_ndc, ndc_max.z) - vec3(min_r_ndc, ndc_min.z));
            rectangle.center = vec3(min_r_ndc, ndc_min.z) + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = vec3(1, 0, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(deref(push.uses.globals).shader_debug_context, rectangle);
        }
    }
    #endif

    return depth_cull;
}

bool get_meshlet_instance_from_arg(uint thread_id, uint arg_bucket_index, daxa_BufferPtr(MeshletCullIndirectArgTable) meshlet_cull_indirect_args, out MeshletInstance meshlet_inst)
{
    const uint indirect_arg_index = thread_id >> arg_bucket_index;
    const uint valid_arg_count = deref(meshlet_cull_indirect_args).indirect_arg_counts[arg_bucket_index];
    if (indirect_arg_index >= valid_arg_count)
    {
        return false;
    }
    const uint arg_work_offset = thread_id - (indirect_arg_index << arg_bucket_index);
    const MeshletCullIndirectArg arg = deref(deref(meshlet_cull_indirect_args).indirect_arg_ptrs[arg_bucket_index][indirect_arg_index]);
    meshlet_inst.entity_index = arg.entity_index;
    meshlet_inst.material_index = arg.material_index;
    meshlet_inst.mesh_index = arg.mesh_index;
    meshlet_inst.meshlet_index = arg.meshlet_indices_offset + arg_work_offset;
    meshlet_inst.in_meshgroup_index = arg.in_meshgroup_index;
    return true;
}