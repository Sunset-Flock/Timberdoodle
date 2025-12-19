#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] AllocatePagesPush allocate_pages_push;

#define AT deref(allocate_pages_push.attachments).attachments

[numthreads(ALLOCATE_PAGES_X_DISPATCH, 1, 1)]
[shader("compute")]
void main(uint3 svdtid : SV_DispatchThreadID)
{
    const int index = (svdtid.z * ALLOCATE_PAGES_X_DISPATCH) + svdtid.x;

    if(index >= AT.vsm_allocation_requests.counter) { return; }

    const AllocationRequest request = AT.vsm_allocation_requests.requests[index];
    const bool is_directional_light = request.mip == -1;
    const bool is_point_light = !is_directional_light && (request.coords.z < VSM_SPOT_LIGHT_OFFSET);
    const bool is_spot_light = !is_directional_light && (request.coords.z >= VSM_SPOT_LIGHT_OFFSET);

    const float4 camera_view_last_row = float4(
        AT.vsm_clip_projections[request.coords.z].camera.view[0][3],
        AT.vsm_clip_projections[request.coords.z].camera.view[1][3],
        AT.vsm_clip_projections[request.coords.z].camera.view[2][3],
        1.0f);

    // If a page is already allocated we are just requesting its redraws (probably due to invalidation).
    // This means we do not need to update any allocations, just need to write the position row of the
    // view matrix with which this page will be later redrawn.
    if(request.already_allocated != 0u)
    {
        if(is_point_light || is_spot_light) { /* NOP - do we want to do something here later? */ }
        else
        {
            AT.vsm_page_view_pos_row.get()[request.coords] = camera_view_last_row;
        }
        return;
    }

    // For not allocated pages we need to search for a free page or free one that is cached but not used this frame.
    // For this we follow the following allocation scheme:
    // - Read the values stored inside FindFreePages header
    //   - If the index is less than free_buffer_counter:
    //     - Read the entry in FreePageBuffer[index]
    //     - Read the entry in AllocationRequest[index]
    //     - Assign new entries to the page_table_texel and meta_memory_texel
    //   - If the (index - free_buffer_counter) < not_visited_buffer_counter:
    //     - Read the entry in NotVisitedPageBuffer[index]
    //     - Read the meta memory entry
    //     - Reset (Deallocate) the entry that previously owned this memory in virtual page table 
    //     - Assign new entries to the page_table_texel and meta_memory_texel

    FindFreePagesHeader * header = AT.vsm_find_free_pages_header;

    const int free_pages_shifted_index = index - header.free_buffer_counter;
    // First try to use up all the free pages.
    if(index < header->free_buffer_counter)
    {
        // Get the coordinates of a free page (in the memory texture) and pack them into a single uint.
        const int2 free_memory_page_coords = AT.vsm_free_pages_buffer[index].coords;
        const uint new_vsm_page_entry = pack_meta_coords_to_vsm_entry(free_memory_page_coords) | allocated_mask();

        // Write this into the respective page table.
        uint64_t packed_meta_memory_page_entry = 0;
        if(is_point_light || is_spot_light)
        {
            AT.vsm_point_spot_page_table[request.mip].get_formatted()[uint3(request.coords)] = new_vsm_page_entry;

            const PointSpotLightCoords vsm_point_spot_light_page_coords = PointSpotLightCoords(
                request.coords.xy,          // texel_coords
                request.mip,                // mip_level
                request.coords.z);          // array_layer_index

            packed_meta_memory_page_entry = pack_vsm_point_spot_light_coords_to_meta_entry(vsm_point_spot_light_page_coords) | meta_memory_point_spot_light_mask();
        }
        else 
        {
            AT.vsm_page_table.get_formatted()[request.coords] = new_vsm_page_entry;
            AT.vsm_page_view_pos_row.get()[request.coords] = camera_view_last_row;
            packed_meta_memory_page_entry = pack_vsm_coords_to_meta_entry(request.coords);

        }

        // Write the packed coordinates of the virtual vsm page into the meta memory entry.
        const uint64_t alloc_packed_meta_memory_page_entry = packed_meta_memory_page_entry | meta_memory_allocated_mask();
        AT.vsm_meta_memory_table.get_formatted()[free_memory_page_coords] = alloc_packed_meta_memory_page_entry;
    }
    // If there are not enough free pages we need to free some cached pages and allocate into them instead.
    else if(free_pages_shifted_index < header->not_visited_buffer_counter)
    {
        // Get meta information about the page owning the previous allocation.
        const int2 not_visited_memory_page_coords = AT.vsm_not_visited_pages_buffer[free_pages_shifted_index].coords;
        const uint64_t meta_entry = AT.vsm_meta_memory_table.get_formatted()[not_visited_memory_page_coords];

        const uint new_vsm_page_entry = pack_meta_coords_to_vsm_entry(not_visited_memory_page_coords) | allocated_mask();

        // Reset previously owning vsm page
        if(get_meta_memory_is_point_spot_light(meta_entry))
        {
            const PointSpotLightCoords owning_vsm_coords = get_vsm_point_spot_light_coords_from_meta_entry(meta_entry);
            const int3 page_coords = int3(owning_vsm_coords.texel_coords, owning_vsm_coords.array_layer_index);
            AT.vsm_point_spot_page_table[owning_vsm_coords.mip_level].get_formatted()[page_coords] = 0u;
        }
        else
        {
            const int3 owning_vsm_coords = get_vsm_coords_from_meta_entry(meta_entry);
            AT.vsm_page_table.get_formatted()[owning_vsm_coords] = 0u;
        }

        uint64_t packed_meta_memory_page_entry = 0;
        if(is_point_light || is_spot_light)
        {
            AT.vsm_point_spot_page_table[request.mip].get_formatted()[uint3(request.coords)] = new_vsm_page_entry;

            const PointSpotLightCoords vsm_point_spot_light_page_coords = PointSpotLightCoords(
                request.coords.xy,          // texel_coords
                request.mip,                // mip_level
                request.coords.z);          // array_layer_index

            packed_meta_memory_page_entry = pack_vsm_point_spot_light_coords_to_meta_entry(vsm_point_spot_light_page_coords) | meta_memory_point_spot_light_mask();
        }
        else 
        {
            AT.vsm_page_table.get_formatted()[request.coords] = new_vsm_page_entry;
            AT.vsm_page_view_pos_row.get()[request.coords] = camera_view_last_row;
            packed_meta_memory_page_entry = pack_vsm_coords_to_meta_entry(request.coords);
        }

        // Write the packed coordinates of the virtual vsm page into the meta memory entry.
        const uint64_t alloc_packed_meta_memory_page_entry = packed_meta_memory_page_entry | meta_memory_allocated_mask();
        AT.vsm_meta_memory_table.get_formatted()[not_visited_memory_page_coords] = alloc_packed_meta_memory_page_entry;
    }
    // If we have no pages to free (ie we ran out of memory) mark this page as allocation failed.
    else
    {
        AT.vsm_page_table.get_formatted()[request.coords] = allocation_failed_mask();
    }
}