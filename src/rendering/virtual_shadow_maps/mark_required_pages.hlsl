#include "daxa/daxa.inl"

#include "vsm.inl"
#include "shader_shared/vsm_shared.inl"
#include "shader_lib/vsm_util.glsl"

[[vk::push_constant]] MarkRequiredPagesH::AttachmentShaderBlob mark_pages_push;

void request_out_of_range_page(const float depth_min, const float depth_max)
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
}

[numthreads(MARK_REQUIRED_PAGES_X_DISPATCH, MARK_REQUIRED_PAGES_Y_DISPATCH, 1)]
[shader("compute")]
void main(uint3 svdtid : SV_DispatchThreadID)
{
    if(all(lessThan(svdtid.xy, mark_pages_push.globals.settings.render_target_size)))
    {
        const float depth = mark_pages_push.depth.get().Load(int3(svdtid.xy, 0)).r;

        // Skip fragments into which no objects were rendered
        if(depth == 0.0f) { return; }

        const float4x4 inv_projection_view = mark_pages_push.globals.camera.inv_view_proj;
        const float2 screen_space_uv = (float2(svdtid.xy) + float2(0.5f)) * mark_pages_push.globals.settings.render_target_size_inv;

        const ClipInfo clip_info = clip_info_from_uvs(ClipFromUVsInfo(
            screen_space_uv,
            mark_pages_push.globals.settings.render_target_size,
            depth,
            inv_projection_view,
            -1,
            mark_pages_push.vsm_clip_projections,
            mark_pages_push.vsm_globals,
            mark_pages_push.globals
        ));
        if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return; }

        const int3 vsm_page_wrapped_coords = vsm_clip_info_to_wrapped_coords(clip_info, mark_pages_push.vsm_clip_projections);
        if(vsm_page_wrapped_coords.x < 0 || vsm_page_wrapped_coords.y < 0) { return; }

        uint prev_page_state;
        bool thread_active = true;
        bool first_to_see = false;

        float sg_min_depth;
        float sg_max_depth;
        while(thread_active)
        {
            const int3 sg_uniform_page_wrapped_coords = WaveReadLaneFirst(vsm_page_wrapped_coords);

            if(all(equal(sg_uniform_page_wrapped_coords, vsm_page_wrapped_coords)))
            {
                sg_min_depth = WaveActiveMin(clip_info.clip_depth);
                sg_max_depth = WaveActiveMax(clip_info.clip_depth);
                if(WaveIsFirstLane())
                {
                    first_to_see = true;
                }
                thread_active = false;
            }
        }

        if(first_to_see)
        {
            InterlockedOr(
                mark_pages_push.vsm_page_table.get()[vsm_page_wrapped_coords],
                uint(requests_allocation_mask() | visited_marked_mask()),
                prev_page_state
            );

            if(!get_requests_allocation(prev_page_state) && !get_is_allocated(prev_page_state))
            {
                uint allocation_index;
                InterlockedAdd(mark_pages_push.vsm_allocation_count->count, 1u, allocation_index);
                if(allocation_index < MAX_VSM_ALLOC_REQUESTS)
                {
                    mark_pages_push.vsm_allocation_requests[allocation_index] = AllocationRequest(vsm_page_wrapped_coords, 0u);
                }
            }
            else if(get_is_allocated(prev_page_state) && !get_is_visited_marked(prev_page_state))
            {
                // TODO(msakmary) finish fix
                request_out_of_range_page(sg_min_depth, sg_max_depth);
                InterlockedOr(
                    mark_pages_push.vsm_meta_memory_table.get()[get_meta_coords_from_vsm_entry(prev_page_state)],
                    meta_memory_visited_mask()
                );
            }
        }
    }
}