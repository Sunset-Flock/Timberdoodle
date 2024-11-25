#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

DAXA_DECL_PUSH_CONSTANT(ClearPagesPush, push)
layout (local_size_x = CLEAR_PAGES_X_DISPATCH, local_size_y = CLEAR_PAGES_Y_DISPATCH) in;
void main()
{
    uint vsm_page_entry = 0;
    const ivec3 alloc_request_page_coords = deref(push.attachments.vsm_allocation_requests).requests[gl_GlobalInvocationID.z].coords;

    // TODO: Point lights
    if(deref(push.attachments.vsm_allocation_requests).requests[gl_GlobalInvocationID.z].point_light_index != -1)
    {
        return;
    }

    if(gl_SubgroupInvocationID == 0)
    {
        vsm_page_entry = imageLoad(daxa_uimage2DArray(push.attachments.vsm_page_table), alloc_request_page_coords).r;
        const uint vsm_page_entry_marked_dirty = vsm_page_entry | dirty_mask();
        imageStore(daxa_uimage2DArray(push.attachments.vsm_page_table), alloc_request_page_coords, daxa_u32vec4(vsm_page_entry_marked_dirty));
    }
    vsm_page_entry = subgroupBroadcast(vsm_page_entry, 0);
    if(!get_is_allocated(vsm_page_entry)) { return; }

    const ivec2 memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
    const ivec2 in_memory_corner_coords = memory_page_coords * VSM_PAGE_SIZE;
    const ivec2 in_memory_workgroup_offset = ivec2(gl_WorkGroupID.xy) * CLEAR_PAGES_X_DISPATCH;
    const ivec2 thread_memory_coords = in_memory_corner_coords + in_memory_workgroup_offset + ivec2(gl_LocalInvocationID.xy);
    imageStore(daxa_image2D(push.attachments.vsm_memory), thread_memory_coords, vec4(1.0));
}