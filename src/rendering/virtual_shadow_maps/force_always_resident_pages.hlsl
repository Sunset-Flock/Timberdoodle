#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] ForceAlwaysResidentPagesH::AttachmentShaderBlob force_pages_push;

[numthreads(FORCE_ALWAYS_PRESENT_PAGES_X_DISPATCH)]
[shader("compute")]
void main(
    uint3 svdtid : SV_DispatchThreadID
    )
{
    const int point_light_array_layer_count = force_pages_push.globals.vsm_settings.point_light_count * 6;
    const int spot_light_array_layer_count = force_pages_push.globals.vsm_settings.spot_light_count;
    const int array_layer_count = point_light_array_layer_count + spot_light_array_layer_count;

    if (svdtid.x < array_layer_count) 
    {
        const bool is_spot_light = svdtid.x >= point_light_array_layer_count;
        const int light_index = svdtid.x - (uint(is_spot_light) * point_light_array_layer_count);
        const int array_layer_index = light_index + (uint(is_spot_light) * VSM_SPOT_LIGHT_OFFSET);

        uint prev_page_state;
        InterlockedOr(
            force_pages_push.vsm_point_spot_page_table.get_formatted()[uint3(0u, 0u, array_layer_index)],
            uint(requests_allocation_mask() | visited_marked_mask()),
            prev_page_state
        );

        if(!get_requests_allocation(prev_page_state) && !get_is_allocated(prev_page_state))
        {
            uint allocation_index;
            InterlockedAdd(force_pages_push.vsm_allocation_requests->counter, 1u, allocation_index);
            if(allocation_index < MAX_VSM_ALLOC_REQUESTS)
            {
                force_pages_push.vsm_allocation_requests.requests[allocation_index] = AllocationRequest(int3(0, 0, array_layer_index), 0u, VSM_FORCED_MIP_LEVEL);
            }
        }
        else if(get_is_allocated(prev_page_state) && !get_is_visited_marked(prev_page_state))
        {
            InterlockedOr(
                force_pages_push.vsm_meta_memory_table.get_formatted()[get_meta_coords_from_vsm_entry(prev_page_state)],
                meta_memory_visited_mask()
            );
        }
    }
}