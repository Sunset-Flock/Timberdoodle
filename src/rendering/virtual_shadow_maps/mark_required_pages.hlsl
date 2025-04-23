#include "daxa/daxa.inl"

#include "vsm.inl"
#include "shader_shared/vsm_shared.inl"
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] MarkRequiredPagesPush mark_pages_push;

#define AT deref(mark_pages_push.attachments).attachments

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
    if(all(lessThan(svdtid.xy, AT.globals.settings.render_target_size)))
    {
        const float depth = AT.g_buffer_depth.get().Load(int3(svdtid.xy, 0)).r;
        const uint compressed_geo_normal = AT.g_buffer_geo_normal.get().Load(int3(svdtid.xy, 0)).r;
        const float3 geo_normal = uncompress_normal_octahedral_32(compressed_geo_normal);

        // Skip fragments into which no objects were rendered
        if(depth == 0.0f) { return; }

        const float4x4 inv_projection_view = AT.globals.camera.inv_view_proj;
        const float2 screen_space_tex_center_uv = (float2(svdtid.xy) + float2(0.5f)) * AT.globals.settings.render_target_size_inv;

        const ClipInfo clip_info = clip_info_from_uvs(ClipFromUVsInfo(
            screen_space_tex_center_uv,
            AT.globals.settings.render_target_size,
            depth,
            inv_projection_view,
            -1,
            AT.vsm_clip_projections,
            AT.vsm_globals,
            AT.globals
        ));
        if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return; }

        const int3 vsm_page_wrapped_coords = vsm_clip_info_to_wrapped_coords(clip_info, AT.vsm_clip_projections);


        uint prev_page_state;
        bool thread_active = (vsm_page_wrapped_coords.x >= 0 && vsm_page_wrapped_coords.y >= 0);
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
                AT.vsm_page_table.get_formatted()[vsm_page_wrapped_coords],
                uint(requests_allocation_mask() | visited_marked_mask()),
                prev_page_state
            );

            if(!get_requests_allocation(prev_page_state) && !get_is_allocated(prev_page_state))
            {
                uint allocation_index;
                InterlockedAdd(AT.vsm_allocation_requests->counter, 1u, allocation_index);
                if(allocation_index < MAX_VSM_ALLOC_REQUESTS)
                {
                    AT.vsm_allocation_requests->requests[allocation_index] = AllocationRequest(vsm_page_wrapped_coords, 0u, -1);
                }
            }
            else if(get_is_allocated(prev_page_state) && !get_is_visited_marked(prev_page_state))
            {
                // TODO(msakmary) finish fix
                request_out_of_range_page(sg_min_depth, sg_max_depth);
                InterlockedOr(
                    AT.vsm_meta_memory_table.get_formatted()[get_meta_coords_from_vsm_entry(prev_page_state)],
                    meta_memory_visited_mask()
                );
            }
        }

        for(int point_light_idx = 0; point_light_idx < AT.globals.vsm_settings.point_light_count; ++point_light_idx)
        {
            const float2 screen_space_uv = float2(svdtid.xy) * AT.globals.settings.render_target_size_inv;

            const PointMipInfo vsm_light_mip_info = project_into_point_light(
                depth,
                geo_normal,
                point_light_idx,
                screen_space_uv,
                AT.globals,
                AT.vsm_point_lights,
                AT.vsm_globals
            );

            bool thread_active = vsm_light_mip_info.cube_face != -1;
            bool first_to_see = false;

            const uint array_layer = get_vsm_point_page_array_idx(vsm_light_mip_info.cube_face, point_light_idx);
            const int4 vsm_point_page_coords = int4(vsm_light_mip_info.page_texel_coords, vsm_light_mip_info.cube_face, vsm_light_mip_info.mip_level);
            while(thread_active)
            {
                const int4 sg_uniform_point_page_coords = WaveReadLaneFirst(vsm_point_page_coords);

                if(all(equal(sg_uniform_point_page_coords, vsm_point_page_coords)))
                {
                    if(WaveIsFirstLane())
                    {
                        first_to_see = true;
                    }
                    thread_active = false;
                }
            }

            if(first_to_see)
            {
                uint prev_page_state_point;
                const uint point_page_array_index = get_vsm_point_page_array_idx(vsm_point_page_coords.z, point_light_idx);
                InterlockedOr(
                    AT.vsm_point_spot_page_table[vsm_point_page_coords.w].get_formatted()[uint3(vsm_point_page_coords.xy, point_page_array_index)], // [mip].get()[x, y, array_layer]
                    uint(requests_allocation_mask() | visited_marked_mask()),
                    prev_page_state_point
                );

                if(!get_requests_allocation(prev_page_state_point) && !get_is_allocated(prev_page_state_point))
                {
                    uint allocation_index;
                    InterlockedAdd(AT.vsm_allocation_requests->counter, 1u, allocation_index);
                    if(allocation_index < MAX_VSM_ALLOC_REQUESTS)
                    {
                        AT.vsm_allocation_requests.requests[allocation_index] = AllocationRequest(int3(vsm_point_page_coords.xy, point_page_array_index), 0u, vsm_point_page_coords.w);
                    }
                }
                else if(get_is_allocated(prev_page_state_point) && !get_is_visited_marked(prev_page_state_point))
                {
                    InterlockedOr(
                        AT.vsm_meta_memory_table.get_formatted()[get_meta_coords_from_vsm_entry(prev_page_state_point)],
                        meta_memory_visited_mask()
                    );
                }
            }
        }

        for(int spot_light_idx = 0; spot_light_idx < AT.globals.vsm_settings.spot_light_count; ++spot_light_idx)
        {
            const float2 screen_space_uv = float2(svdtid.xy) * AT.globals.settings.render_target_size_inv;

            const SpotMipInfo vsm_light_mip_info = project_into_spot_light(
                depth,
                geo_normal,
                spot_light_idx,
                screen_space_uv,
                AT.globals,
                AT.vsm_spot_lights,
                AT.vsm_globals
            );

            bool thread_active = vsm_light_mip_info.page_texel_coords.x != -1;
            bool first_to_see = false;

            const int3 vsm_spot_page_coords = int3(vsm_light_mip_info.page_texel_coords, vsm_light_mip_info.mip_level);
            while(thread_active)
            {
                const int3 sg_uniform_spot_page_coords = WaveReadLaneFirst(vsm_spot_page_coords);

                if(all(equal(sg_uniform_spot_page_coords, vsm_spot_page_coords)))
                {
                    if(WaveIsFirstLane())
                    {
                        first_to_see = true;
                    }
                    thread_active = false;
                }
            }

            if(first_to_see)
            {
                const uint array_layer_index = VSM_SPOT_LIGHT_OFFSET + spot_light_idx;
                uint prev_page_state_spot;
                // .z is the mip level
                // .xy are the coordinates
                InterlockedOr(
                    AT.vsm_point_spot_page_table[vsm_spot_page_coords.z].get_formatted()[uint3(vsm_spot_page_coords.xy, array_layer_index)],
                    uint(requests_allocation_mask() | visited_marked_mask()),
                    prev_page_state_spot
                );

                if(!get_requests_allocation(prev_page_state_spot) && !get_is_allocated(prev_page_state_spot))
                {
                    uint allocation_index;
                    InterlockedAdd(AT.vsm_allocation_requests->counter, 1u, allocation_index);
                    if(allocation_index < MAX_VSM_ALLOC_REQUESTS)
                    {
                        AT.vsm_allocation_requests.requests[allocation_index] = AllocationRequest(int3(vsm_spot_page_coords.xy, array_layer_index), 0u, vsm_spot_page_coords.z);
                    }
                }
                else if(get_is_allocated(prev_page_state_spot) && !get_is_visited_marked(prev_page_state_spot))
                {
                    InterlockedOr(
                        AT.vsm_meta_memory_table.get_formatted()[get_meta_coords_from_vsm_entry(prev_page_state_spot)],
                        meta_memory_visited_mask()
                    );
                }
            }
        }
    }
}