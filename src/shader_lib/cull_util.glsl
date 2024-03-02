#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"
#include "../shader_shared/cull_util.inl"
#include "../shader_lib/debug.glsl"
#include "../shader_shared/globals.inl"
#include "../shader_shared/geometry.inl"
#include "../shader_shared/geometry_pipeline.inl"

bool is_sphere_out_of_frustum(CameraInfo camera, daxa_f32vec3 ws_center, float ws_radius)
{
    const daxa_f32vec3 frustum_planes[5] = {
        camera.right_plane_normal,
        camera.left_plane_normal,
        camera.top_plane_normal,
        camera.bottom_plane_normal,
        camera.near_plane_normal,
    };
    bool out_of_frustum = false;
    for (uint i = 0; i < 5; ++i)
    {
        out_of_frustum = out_of_frustum || (dot((ws_center - camera.position), frustum_planes[i]) - ws_radius) > 0.0f;
    }
    return out_of_frustum;
}

bool is_tri_out_of_frustum(CameraInfo camera, vec3 tri[3])
{
    const vec3 frustum_planes[5] = {
        camera.right_plane_normal,
        camera.left_plane_normal,
        camera.top_plane_normal,
        camera.bottom_plane_normal,
        camera.near_plane_normal,
    };
    bool out_of_frustum = false;
    for (uint i = 0; i < 5; ++i)
    {
        bool tri_out_of_plane = true;
        for (uint ti = 0; ti < 3; ++ti)
        {
            tri_out_of_plane = tri_out_of_plane && dot((tri[ti] - camera.position), frustum_planes[i]) > 0.0f;
        }
        out_of_frustum = out_of_frustum || tri_out_of_plane;
    }
    return out_of_frustum;
}

bool is_meshlet_occluded(
    CameraInfo camera,
    MeshletInstance meshlet_inst,
    daxa_BufferPtr(daxa_u32) first_pass_meshlets_bitfield_offsets,
    U32ArenaBufferRef first_pass_meshlets_bitfield_arena,
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

    const uint first_pass_meshgroup_bitfield_offset = deref(first_pass_meshlets_bitfield_offsets[meshlet_inst.entity_index]);
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
            DEBUG_INDEX(
                mesh_instance_bitfield_offset_offset,
                0, 
                first_pass_meshlets_bitfield_arena.offsets_section_size - 1);
            if (meshlet_drawn_first_pass)
            {
                return true;
            }
        }
    }

    // daxa_f32vec3 center;
    // daxa_f32 radius;
    mat4x4 model_matrix = mat_4x3_to_4x4(deref(entity_combined_transforms[meshlet_inst.entity_index]));
    mat4x4 view_proj = camera.view_proj;
    const float model_scaling_x_squared = dot(model_matrix[0],model_matrix[0]);
    const float model_scaling_y_squared = dot(model_matrix[1],model_matrix[1]);
    const float model_scaling_z_squared = dot(model_matrix[2],model_matrix[2]);
    const float radius_scaling = sqrt(max(max(model_scaling_x_squared,model_scaling_y_squared), model_scaling_z_squared));
    BoundingSphere bounds = deref(mesh_data.meshlet_bounds[meshlet_inst.meshlet_index]);
    AABB meshlet_aabb = deref(mesh_data.meshlet_aabbs[meshlet_inst.meshlet_index]);
    const float scaled_radius = radius_scaling * bounds.radius;
    const vec3 ws_center = (model_matrix * vec4(bounds.center, 1)).xyz;

    if (is_sphere_out_of_frustum(camera, ws_center, scaled_radius))
    {
        #if defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus)
            ShaderDebugCircleDraw circle;
            circle.position = ws_center;
            circle.radius = scaled_radius;
            circle.color = vec3(1,1,0);
            circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            debug_draw_circle(GLOBALS.debug, circle);
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

    const vec2 f_hiz_resolution = daxa_f32vec2(camera.screen_size >> 1 /*hiz is half res*/);
    const vec2 min_uv = (ndc_min.xy + 1.0f) * 0.5f;
    const vec2 max_uv = (ndc_max.xy + 1.0f) * 0.5f;
    const vec2 min_texel_i = floor(clamp(f_hiz_resolution * min_uv, daxa_f32vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const vec2 max_texel_i = floor(clamp(f_hiz_resolution * max_uv, daxa_f32vec2(0.0f, 0.0f), f_hiz_resolution - 1.0f));
    const float pixel_range = max(max_texel_i.x - min_texel_i.x + 1.0f, max_texel_i.y - min_texel_i.y + 1.0f);
    const float mip = ceil(log2(max(2.0f, pixel_range))) - 1 /* we want one mip lower, as we sample a quad */;

    // The calculation above gives us a mip level, in which the a 2x2 quad in that mip is just large enough to fit the ndc bounds.
    // When the ndc bounds are shofted from the alignment of that mip levels grid however, we need an even larger quad.
    // We check if the quad at its current position within that mip level fits that quad and if not we move up one mip.
    // This will give us the tightest fit.
    int imip = int(mip);
    const ivec2 min_corner_texel = daxa_i32vec2(min_texel_i) >> imip;
    const ivec2 max_corner_texel = daxa_i32vec2(max_texel_i) >> imip;
    if (any(greaterThan(max_corner_texel - min_corner_texel, daxa_i32vec2(1)))) {
        imip += 1;
    }
    const ivec2 quad_corner_texel = daxa_i32vec2(min_texel_i) >> imip;
    const ivec2 texel_bounds = max(daxa_i32vec2(0,0), (daxa_i32vec2(f_hiz_resolution) >> imip) - 1);

    const vec4 fetch = daxa_f32vec4(
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(0,0), daxa_i32vec2(0,0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(0,1), daxa_i32vec2(0,0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(1,0), daxa_i32vec2(0,0), texel_bounds), imip).x,
        texelFetch(daxa_texture2D(hiz), clamp(quad_corner_texel + daxa_i32vec2(1,1), daxa_i32vec2(0,0), texel_bounds), imip).x
    );
    const float conservative_depth = min(min(fetch.x,fetch.y), min(fetch.z, fetch.w));
    const bool depth_cull = ndc_max.z < conservative_depth;

    #if (defined(GLOBALS) && CULLING_DEBUG_DRAWS || defined(__cplusplus)) && false
    if (depth_cull)
    {
        ShaderDebugAABBDraw aabb1;
        aabb1.position = (model_matrix * vec4(meshlet_aabb.center,1)).xyz;
        aabb1.size = (model_matrix * vec4(meshlet_aabb.size,0)).xyz;
        aabb1.color = vec3(0.1, 0.5, 1);
        aabb1.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        debug_draw_aabb(GLOBALS.debug, aabb1);
        {
            ShaderDebugRectangleDraw rectangle;
            const vec3 rec_size = (ndc_max - ndc_min);
            rectangle.center = ndc_min + (rec_size * 0.5);
            rectangle.span = rec_size.xy;
            rectangle.color = vec3(0, 1, 1);
            rectangle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC;
            debug_draw_rectangle(GLOBALS.debug, rectangle);
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
            debug_draw_rectangle(GLOBALS.debug, rectangle);
        }
    }
    #endif

    return depth_cull;
}

// How does this work?
// - this is an asymertric work distribution problem
// - each mesh cull thread needs x followup threads where x is the number of meshlets for the mesh
// - writing x times to some argument buffer to dispatch over later is extreamly divergent and inefficient
//   - solution is to combine writeouts in powers of two:
//   - instead of x writeouts, only do log2(x), one writeout per set bit in the meshletcount.
//   - when you want to write out 17 meshlet work units, instead of writing 7 args into a buffer,
//     you write one 1x arg, no 2x arg, no 4x arg, no 8x arg and one 16x arg. the 1x and the 16x args together contain 17 work units.
// - still not good enough, in large cases like 2^16 - 1 meshlets it would need 15 writeouts, that is too much!
//   - solution is to limit the writeouts to some smaller number (i chose 5, as it has a max thread waste of < 5%)
//   - A strong compromise is to round up invocation count from meshletcount in such a way that the round up value only has 4 bits set at most.
//   - as we do one writeout per bit set in meshlet count, this limits the writeout to 5.
// - in worst case this can go down from thousands of divergent writeouts down to 5 while only wasting < 5% of invocations.
void write_meshlet_cull_arg_buckets(
    GPUMesh mesh,
    const MeshDrawTuple mesh_draw,
    daxa_RWBufferPtr(MeshletCullArgBucketsBufferHead) cull_buckets,
    const uint meshlet_cull_shader_workgroup_x,
    const uint cull_shader_workgroup_log2)
{
    const uint MAX_BITS = 5;
    uint meshlet_count_msb = findMSB(mesh.meshlet_count);
    const uint shift = uint(max(0, int(meshlet_count_msb) + 1 - int(MAX_BITS)));
    // clip off all bits below the 5 most significant ones.
    uint clipped_bits_meshlet_count = (mesh.meshlet_count >> shift) << shift;
    // Need to round up if there were bits clipped.
    if (clipped_bits_meshlet_count < mesh.meshlet_count)
    {
        clipped_bits_meshlet_count += (1 << shift);
    }
    // Now bit by bit, do one writeout of an indirect command:
    uint bucket_bit_mask = clipped_bits_meshlet_count;
    // Each time we write out a command we add on the number of meshlets processed by that arg.
    uint meshlet_offset = 0;
    while (bucket_bit_mask != 0)
    {
        const uint bucket_index = findMSB(bucket_bit_mask);
        const uint indirect_arg_meshlet_count = 1 << (bucket_index);
        // Mask out bit.
        bucket_bit_mask &= ~indirect_arg_meshlet_count;
        const uint arg_array_offset = atomicAdd(deref(cull_buckets).indirect_arg_counts[bucket_index], 1);
        // Update indirect args for meshlet cull
        {
            const uint threads_per_indirect_arg = 1 << bucket_index;
            const uint prev_indirect_arg_count = arg_array_offset;
            const uint prev_needed_threads = threads_per_indirect_arg * prev_indirect_arg_count;
            const uint prev_needed_workgroups = (prev_needed_threads + meshlet_cull_shader_workgroup_x - 1) >> cull_shader_workgroup_log2;
            const uint cur_indirect_arg_count = arg_array_offset + 1;
            const uint cur_needed_threads = threads_per_indirect_arg * cur_indirect_arg_count;
            const uint cur_needed_workgroups = (cur_needed_threads + meshlet_cull_shader_workgroup_x - 1) >> cull_shader_workgroup_log2;

            const bool update_cull_meshlets_dispatch = prev_needed_workgroups != cur_needed_workgroups;
            if (update_cull_meshlets_dispatch)
            {
                atomicMax(deref(cull_buckets).commands[bucket_index].x, cur_needed_workgroups);
            }
        }
        MeshletCullIndirectArg arg;
        arg.entity_index = mesh_draw.entity_index;
        arg.mesh_index = mesh_draw.mesh_index;
        arg.material_index = mesh.material_index;
        arg.in_mesh_group_index = mesh_draw.in_mesh_group_index;
        arg.meshlet_indices_offset = meshlet_offset;
        arg.meshlet_count = min(mesh.meshlet_count - meshlet_offset, indirect_arg_meshlet_count);
        deref(deref(cull_buckets).indirect_arg_ptrs[bucket_index][arg_array_offset]) = arg;
        meshlet_offset += indirect_arg_meshlet_count;
    }
}

bool get_meshlet_instance_from_arg_buckets(uint thread_id, uint arg_bucket_index, daxa_BufferPtr(MeshletCullArgBucketsBufferHead) meshlet_cull_indirect_args, out MeshletInstance meshlet_inst)
{
    const uint indirect_arg_index = thread_id >> arg_bucket_index;
    const uint valid_arg_count = deref(meshlet_cull_indirect_args).indirect_arg_counts[arg_bucket_index];
    // As work groups are launched in multiples of 128 (or 32 in the case of task shaders), 
    // there may be threads with indices greater then the arg count for a bucket.
    if (indirect_arg_index >= valid_arg_count)
    {
        return false;
    }
    const uint in_arg_meshlet_index = thread_id - (indirect_arg_index << arg_bucket_index);
    const MeshletCullIndirectArg arg = deref(deref(meshlet_cull_indirect_args).indirect_arg_ptrs[arg_bucket_index][indirect_arg_index]);
    // Work argument may work on less then 1<<bucket_index meshlets.
    // In this case we cull threads with an index over meshlet_count.
    if (in_arg_meshlet_index >= arg.meshlet_count)
    {
        return false;
    }
    meshlet_inst.entity_index = arg.entity_index;
    meshlet_inst.material_index = arg.material_index;
    meshlet_inst.mesh_index = arg.mesh_index;
    meshlet_inst.meshlet_index = arg.meshlet_indices_offset + in_arg_meshlet_index;
    meshlet_inst.in_mesh_group_index = arg.in_mesh_group_index;
    return true;
}