#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"
#include "../shader_shared/asset.inl"

struct NdcBounds
{
    vec3 ndc_min;
    vec3 ndc_max;
    uint valid_vertices;
};

void init_ndc_bounds(inout NdcBounds ndc_bounds)
{
    ndc_bounds.ndc_min = vec3(0);
    ndc_bounds.ndc_max = vec3(0);
    ndc_bounds.valid_vertices = 0;
}

// All vertex positions MUST be in front of the near plane!
void add_vertex_to_ndc_bounds(inout NdcBounds ndc_bounds, vec3 ndc_pos)
{
    if (ndc_bounds.valid_vertices == 0)
    {
        ndc_bounds.ndc_min = ndc_pos;
        ndc_bounds.ndc_max = ndc_pos;
    }
    else
    {
        ndc_bounds.ndc_min = vec3(
            min(ndc_pos.x, ndc_bounds.ndc_min.x),
            min(ndc_pos.y, ndc_bounds.ndc_min.y),
            min(ndc_pos.z, ndc_bounds.ndc_min.z)
        );
        ndc_bounds.ndc_max = vec3(
            max(ndc_pos.x, ndc_bounds.ndc_max.x),
            max(ndc_pos.y, ndc_bounds.ndc_max.y),
            max(ndc_pos.z, ndc_bounds.ndc_max.z)
        );
    }
    ndc_bounds.valid_vertices += 1;
}

mat4 mat_4x3_to_4x4(mat4x3 in_mat)
{
    return mat4(
        vec4(in_mat[0], 0.0),
        vec4(in_mat[1], 0.0),
        vec4(in_mat[2], 0.0),
        vec4(in_mat[3], 1.0)
    );
}

bool is_out_of_frustum(vec3 ws_center, float ws_radius)
{
    const vec3 frustum_planes[5] = {
        globals.camera_right_plane_normal,
        globals.camera_left_plane_normal,
        globals.camera_top_plane_normal,
        globals.camera_bottom_plane_normal,
        globals.camera_near_plane_normal,
    };
    bool out_of_frustum = false;
    for (uint i = 0; i < 5; ++i)
    {
        out_of_frustum = out_of_frustum || (dot((ws_center - globals.camera_pos), frustum_planes[i]) - ws_radius) > 0.0f;
    }
    return out_of_frustum;
}

bool is_tri_out_of_frustum(vec3 tri[3])
{
    const vec3 frustum_planes[5] = {
        globals.camera_right_plane_normal,
        globals.camera_left_plane_normal,
        globals.camera_top_plane_normal,
        globals.camera_bottom_plane_normal,
        globals.camera_near_plane_normal,
    };
    bool out_of_frustum = false;
    for (uint i = 0; i < 5; ++i)
    {
        bool tri_out_of_plane = true;
        for (uint ti = 0; ti < 3; ++ti)
        {
            tri_out_of_plane = tri_out_of_plane && dot((tri[ti] - globals.camera_pos), frustum_planes[i]) > 0.0f;
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
    const float model_scaling_x_squared = dot(model_matrix[0],model_matrix[0]);
    const float model_scaling_y_squared = dot(model_matrix[1],model_matrix[1]);
    const float model_scaling_z_squared = dot(model_matrix[2],model_matrix[2]);
    const float radius_scaling = sqrt(max(max(model_scaling_x_squared,model_scaling_y_squared), model_scaling_z_squared));
    BoundingSphere bounds = deref(mesh_data.meshlet_bounds[meshlet_inst.meshlet_index]);
    const float scaled_radius = radius_scaling * bounds.radius;
    const vec3 ws_center = (model_matrix * vec4(bounds.center, 1)).xyz;
    const vec3 center_to_camera = normalize(globals.camera_pos - ws_center);
    const vec3 tangential_up = normalize(globals.camera_up - center_to_camera * dot(center_to_camera, globals.camera_up));
    const vec3 tangent_left = -cross(tangential_up, center_to_camera);
    NdcBounds ndc_bounds;
    init_ndc_bounds(ndc_bounds);
    // construct bounding box from bounding sphere,
    // project each vertex of the box to ndc, min and max the coordinates.
    for (int z = -1; z <= 1; z += 2)
    {
        for (int y = -1; y <= 1; y += 2)
        {
            for (int x = -1; x <= 1; x += 2)
            {
                // TODO: make this use a precalculated obb, not this shit sphere derived one.
                const vec3 bounding_box_corner_ws = bounds.center + bounds.radius * 0.5f * (center_to_camera * z + tangential_up * y + tangent_left * x);
                const vec4 projected_pos = globals.camera_view_projection * model_matrix * vec4(bounding_box_corner_ws, 1);
                const vec3 ndc_pos = projected_pos.xyz / projected_pos.w;
                add_vertex_to_ndc_bounds(ndc_bounds, ndc_pos);
            }
        }
    }
    if (is_out_of_frustum(ws_center, scaled_radius))
    {
        return true;
    }

    if (ndc_bounds.ndc_min.z < 1.0f && ndc_bounds.ndc_min.z > 0.0f)
    {
        const vec2 f_hiz_resolution = vec2(globals.settings.render_target_size >> 1 /*hiz is half res*/);
        const vec2 min_texel_i = floor(clamp(f_hiz_resolution * (ndc_bounds.ndc_min.xy + 1.0f) * 0.5f, vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
        const vec2 max_texel_i = floor(clamp(f_hiz_resolution * (ndc_bounds.ndc_max.xy + 1.0f) * 0.5f, vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
        const float pixel_range = max(max_texel_i.x - min_texel_i.x + 1.0f, max_texel_i.y - min_texel_i.y + 1.0f);
        const float half_pixel_range = max(1.0f, pixel_range * 0.5f /* we will read a area 2x2 */);
        const float mip = ceil(log2(half_pixel_range));

        const ivec2 quad_corner_texel = ivec2(min_texel_i) >> uint(mip);
        const int imip = int(mip);
        const ivec2 mip_size = max(ivec2(0,0),ivec2(globals.settings.render_target_size >> (1 + imip)) - 1);

        const vec4 fetch = vec4(
            texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(0,0), ivec2(0,0), mip_size), int(mip)).x,
            texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(0,1), ivec2(0,0), mip_size), int(mip)).x,
            texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(1,0), ivec2(0,0), mip_size), int(mip)).x,
            texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + ivec2(1,1), ivec2(0,0), mip_size), int(mip)).x
        );
        const float depth = min(min(fetch.x,fetch.y), min(fetch.z, fetch.w));
        const bool depth_cull = (depth > ndc_bounds.ndc_max.z);
        if (depth_cull)
        {
            return true;
        }
    }
    return false;
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