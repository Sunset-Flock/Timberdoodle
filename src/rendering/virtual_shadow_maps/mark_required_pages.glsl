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
        while(active_thread)
        {
            const ivec3 sg_uniform_page_wrapped_coords = subgroupBroadcastFirst(vsm_page_wrapped_coords);

            if(all(equal(sg_uniform_page_wrapped_coords, vsm_page_wrapped_coords)))
            {
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
                    deref_i(push.vsm_allocation_requests, idx) = AllocationRequest(vsm_page_wrapped_coords);
                }
            }
            else if(get_is_allocated(prev_page_state) && !get_is_visited_marked(prev_page_state))
            {
                imageAtomicOr(
                    daxa_access(r32uiImage, push.vsm_meta_memory_table),
                    get_meta_coords_from_vsm_entry(prev_page_state),
                    meta_memory_visited_mask()
                );
            }
        }
    }
}