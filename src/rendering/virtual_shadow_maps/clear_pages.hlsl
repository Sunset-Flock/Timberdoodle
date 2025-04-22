#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] ClearPagesH::AttachmentShaderBlob clear_push;

[numthreads(CLEAR_PAGES_X_DISPATCH, CLEAR_PAGES_Y_DISPATCH, 1)]
[shader("compute")]
void main( 
    uint3 svdtid : SV_DispatchThreadID,
    uint3 svgid : SV_GroupID,
    uint3 svgtid : SV_GroupThreadID
    )
{
    let alloc_request = clear_push.vsm_allocation_requests.requests[svdtid.z];
    uint vsm_page_entry = 0;
    float clear_value;

    if(WaveGetLaneIndex() == 0)
    {
        if(alloc_request.mip != -1)
        {
            // We are clearing a point light page.
            vsm_page_entry = clear_push.vsm_point_spot_page_table[alloc_request.mip].get()[uint3(alloc_request.coords)];
            const uint vsm_page_entry_marked_dirty = vsm_page_entry | dirty_mask();
            clear_push.vsm_point_spot_page_table[alloc_request.mip].get()[uint3(alloc_request.coords)] = vsm_page_entry_marked_dirty;
            clear_value = 0.0f;
        }
        else 
        {
            // We are clearing a directional light source page.
            vsm_page_entry = clear_push.vsm_page_table.get()[alloc_request.coords];
            const uint vsm_page_entry_marked_dirty = vsm_page_entry | dirty_mask();
            clear_push.vsm_page_table.get()[alloc_request.coords] = vsm_page_entry_marked_dirty;
            clear_value = 1.0f;
        }
    }

    vsm_page_entry = WaveBroadcastLaneAt(vsm_page_entry, 0);
    clear_value = WaveBroadcastLaneAt(clear_value, 0);
    if(!get_is_allocated(vsm_page_entry)) { return; }

    const int2 memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);
    const int2 in_memory_corner_coords = memory_page_coords * VSM_PAGE_SIZE;
    const int2 in_memory_workgroup_offset = svgid.xy * CLEAR_PAGES_X_DISPATCH;
    const uint2 thread_memory_coords = in_memory_corner_coords + in_memory_workgroup_offset + svgtid.xy;
    clear_push.vsm_memory_block.get()[thread_memory_coords] = clear_value;
}