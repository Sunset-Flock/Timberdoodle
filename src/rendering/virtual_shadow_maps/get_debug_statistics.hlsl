#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] GetDebugStatisticsH::AttachmentShaderBlob get_debug_statistics_push;

[numthreads(GET_DEBUG_STATISTICS_X_DISPATCH, GET_DEBUG_STATISTICS_Y_DISPATCH, 1)]
[shader("compute")]
void main(uint3 svdtid : SV_DispatchThreadID)
{
    let push = get_debug_statistics_push;

    if(svdtid.x == 0)
    {
        const FindFreePagesHeader * header = push.vsm_find_free_pages_header;
        push.globals.readback.cached_pages = header.not_visited_buffer_counter;
        push.globals.readback.free_pages = header.free_buffer_counter;
        push.globals.readback.drawn_pages = push.vsm_allocation_requests.counter;
    }

    if(all(lessThan(svdtid.xy, int2(VSM_META_MEMORY_TABLE_RESOLUTION, VSM_META_MEMORY_TABLE_RESOLUTION))))
    {
        const uint64_t meta_entry = push.vsm_meta_memory_table.get_formatted()[svdtid.xy];
        // Allocated and Visited -> Cached visible
        if(get_meta_memory_is_allocated(meta_entry))
        {
            if(get_meta_memory_is_point_spot_light(meta_entry))
            {
                if(get_meta_memory_is_visited(meta_entry))
                {
                    InterlockedAdd(push.globals.readback.point_spot_cached_visible_pages, 1);
                }
                else
                {
                    InterlockedAdd(push.globals.readback.point_spot_cached_pages, 1);
                }
            }
            else
            {
                if(get_meta_memory_is_visited(meta_entry))
                {
                    InterlockedAdd(push.globals.readback.directional_cached_visible_pages, 1);
                }
                else
                {
                    InterlockedAdd(push.globals.readback.directional_cached_pages, 1);
                }
            }
        }
    }

    if(svdtid.x < push.vsm_allocation_requests.counter) 
    {
        let allocation_request = push.vsm_allocation_requests.requests[svdtid.x];
        const bool is_directional = (allocation_request.mip == -1);
        if(is_directional) { InterlockedAdd(push.globals.readback.drawn_directional_pages, 1); }
        else               { InterlockedAdd(push.globals.readback.drawn_point_spot_pages, 1);}
    }
}