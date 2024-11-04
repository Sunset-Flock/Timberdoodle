#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

#extension GL_EXT_debug_printf : enable

DAXA_DECL_IMAGE_ACCESSOR_WITH_FORMAT(uimage2DArray, r32ui, , r32uiImageArray)
DAXA_DECL_IMAGE_ACCESSOR_WITH_FORMAT(uimage2D, r32ui, , r32uiImage)

// For each fragment check if the page that will be needed during the shadowmap test is allocated
// if not mark page as needing allocation
DAXA_DECL_PUSH_CONSTANT(MarkRequiredPagesH, push)
layout(local_size_x = MARK_REQUIRED_PAGES_X_DISPATCH, local_size_y = MARK_REQUIRED_PAGES_Y_DISPATCH) in;
void main()
{
    // Depth buffer size should match render target size
    uvec2 render_target_size = deref(push.globals).settings.render_target_size;
    if(all(lessThan(gl_GlobalInvocationID.xy, render_target_size)))
    {
        const float depth = texelFetch(daxa_texture2D(push.depth), ivec2(gl_GlobalInvocationID.xy), 0).r;
        // Skip fragments into which no objects were rendered
        if(depth == 0.0) { return; }

        const mat4x4 inv_projection_view = deref(push.globals).camera.inv_view_proj;
        const vec2 screen_space_uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) / vec2(render_target_size);

        ClipInfo clip_info = clip_info_from_uvs(ClipFromUVsInfo(
            screen_space_uv,
            render_target_size,
            depth,
            inv_projection_view,
            -1,
            push.vsm_clip_projections,
            push.vsm_globals,
            push.globals
        ));
        if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return; }

        const ivec3 vsm_page_wrapped_coords = vsm_clip_info_to_wrapped_coords(clip_info, push.vsm_clip_projections);
        if(vsm_page_wrapped_coords.x < 0 || vsm_page_wrapped_coords.y < 0) { return; }

        uint prev_page_state;
        bool active_thread = true;
        bool first_to_see = false;
        float sg_min_depth;
        float sg_max_depth;
        while(active_thread)
        {
            const ivec3 sg_uniform_page_wrapped_coords = subgroupBroadcastFirst(vsm_page_wrapped_coords);

            if(all(equal(sg_uniform_page_wrapped_coords, vsm_page_wrapped_coords)))
            {
                sg_min_depth = subgroupMin(clip_info.clip_depth);
                sg_max_depth = subgroupMax(clip_info.clip_depth);
                if(subgroupElect())
                {
                    first_to_see = true;
                }
                active_thread = false;
            }
        }

        if(first_to_see)
        {
            prev_page_state = imageAtomicOr(
                daxa_access(r32uiImageArray, push.vsm_page_table),
                vsm_page_wrapped_coords,
                requests_allocation_mask() | visited_marked_mask()
            );

            if(!get_requests_allocation(prev_page_state) && !get_is_allocated(prev_page_state))
            {
                uint idx = atomicAdd(deref(push.vsm_allocation_count).count, 1);
                if(idx < MAX_VSM_ALLOC_REQUESTS)
                {
                    deref_i(push.vsm_allocation_requests, idx) = AllocationRequest(vsm_page_wrapped_coords, 0);
                }
            }
            else if(get_is_allocated(prev_page_state) && !get_is_visited_marked(prev_page_state))
            {
                // TODO(msakmary) This is still broken WHY??
                // if(!get_is_dirty(prev_page_state))
                // {
                //     const vec3 page_view_pos_row = imageLoad(daxa_image2DArray(push.vsm_page_view_pos_row), vsm_page_wrapped_coords).xyz;

                //     mat4x4 page_view_matrix = deref_i(push.vsm_clip_projections, clip_info.clip_level).camera.view;
                //     page_view_matrix[3] = vec4(page_view_pos_row, 1.0f);

                //     const vec4 min_ndc_pos = vec4(0.0f, 0.0f, sg_min_depth, 1.0);
                //     const vec4 max_ndc_pos = vec4(0.0f, 0.0f, sg_max_depth, 1.0);
                //     const vec4 min_ws_pos = deref_i(push.vsm_clip_projections, clip_info.clip_level).camera.inv_view_proj * min_ndc_pos;
                //     const vec4 max_ws_pos = deref_i(push.vsm_clip_projections, clip_info.clip_level).camera.inv_view_proj * max_ndc_pos;

                //     const float min_page_vs_dist = -(page_view_matrix * min_ws_pos).z;
                //     const float max_page_vs_dist = -(page_view_matrix * max_ws_pos).z;

                //     const float near_dist = deref_i(push.vsm_clip_projections, clip_info.clip_level).near_dist;
                //     const float near_to_far_range = deref_i(push.vsm_clip_projections, clip_info.clip_level).near_to_far_range;
                //     const float bias = 0.1 * pow(2.0f, vsm_page_wrapped_coords.z);

                //     const vec2 valid_page_vs_range = vec2(near_dist + bias, near_dist + near_to_far_range + bias);
                //     if(min_page_vs_dist < valid_page_vs_range.x || max_page_vs_dist > valid_page_vs_range.y)
                //     {
                //         uint dirty_state = imageAtomicOr(
                //             daxa_access(r32uiImageArray, push.vsm_page_table),
                //             vsm_page_wrapped_coords,
                //             dirty_mask()
                //         );
                //         if(!get_is_dirty(dirty_state))
                //         {
                //             uint idx = atomicAdd(deref(push.vsm_allocation_count).count, 1);
                //             if(idx < MAX_VSM_ALLOC_REQUESTS)
                //             {
                //                 deref_i(push.vsm_allocation_requests, idx) = AllocationRequest(vsm_page_wrapped_coords, 1);
                //             }
                //         }
                //     }
                // }
                imageAtomicOr(
                    daxa_access(r32uiImage, push.vsm_meta_memory_table),
                    get_meta_coords_from_vsm_entry(prev_page_state),
                    meta_memory_visited_mask()
                );
            }
        }
    }
}