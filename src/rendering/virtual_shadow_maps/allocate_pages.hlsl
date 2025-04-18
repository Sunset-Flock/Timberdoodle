#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

[[vk::push_constant]] AllocatePagesH::AttachmentShaderBlob allocate_pages_push;

[numthreads(ALLOCATE_PAGES_X_DISPATCH, 1, 1)]
[shader("compute")]
void main(uint3 svdtid : SV_DispatchThreadID)
{
    const int index = (svdtid.z * ALLOCATE_PAGES_X_DISPATCH) + svdtid.x;

    if(index >= allocate_pages_push.vsm_allocation_requests.counter) { return; }

    const AllocationRequest request = allocate_pages_push.vsm_allocation_requests.requests[index];
    const bool is_point_light = request.point_light_index != -1;

    const float4 camera_view_last_row = float4(
        allocate_pages_push.vsm_clip_projections[request.coords.z].camera.view[0][3],
        allocate_pages_push.vsm_clip_projections[request.coords.z].camera.view[1][3],
        allocate_pages_push.vsm_clip_projections[request.coords.z].camera.view[2][3],
        1.0f);

    // If a page is already allocated we are just requesting its redraws (probably due to invalidation).
    // This means we do not need to update any allocations, just need to write the position row of the
    // view matrix with which this page will be later redrawn.
    if(request.already_allocated != 0u)
    {
        if(is_point_light) { /* NOP - do we want to do something here later? */ }
        else
        {
            allocate_pages_push.vsm_page_view_pos_row.get()[request.coords] = camera_view_last_row;
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

    const FindFreePagesHeader * header = allocate_pages_push.vsm_find_free_pages_header;

    const int free_pages_shifted_index = index - header.free_buffer_counter;
    // First try to use up all the free pages.
    if(index < header->free_buffer_counter)
    {
        // Get the coordinates of a free page (in the memory texture) and pack them into a single uint.
        const int2 free_memory_page_coords = allocate_pages_push.vsm_free_pages_buffer[index].coords;
        const uint new_vsm_page_entry = pack_meta_coords_to_vsm_entry(free_memory_page_coords) | allocated_mask();

        // Write this into the respective page table.
        uint64_t packed_meta_memory_page_entry = 0;
        if(is_point_light)
        {
            const uint point_page_array_index = get_vsm_point_page_array_idx(request.coords.z, request.point_light_index);
            allocate_pages_push.vsm_point_page_table[request.point_light_mip].get()[uint3(request.coords.xy, point_page_array_index)] = new_vsm_page_entry;

            const PointLightCoords vsm_point_light_page_coords = PointLightCoords(
                request.coords.xy,          // texel_coords
                request.point_light_mip,    // mip_level
                request.coords.z,           // face_index
                request.point_light_index); // point_light_index

            packed_meta_memory_page_entry = pack_vsm_point_light_coords_to_meta_entry(vsm_point_light_page_coords) | meta_memory_point_light_mask();
        }
        else 
        {
            allocate_pages_push.vsm_page_table.get()[request.coords] = new_vsm_page_entry;
            allocate_pages_push.vsm_page_view_pos_row.get()[request.coords] = camera_view_last_row;
            packed_meta_memory_page_entry = pack_vsm_coords_to_meta_entry(request.coords);

        }

        // Write the packed coordinates of the virtual vsm page into the meta memory entry.
        const uint64_t alloc_packed_meta_memory_page_entry = packed_meta_memory_page_entry | meta_memory_allocated_mask();
        allocate_pages_push.vsm_meta_memory_table.get()[free_memory_page_coords] = alloc_packed_meta_memory_page_entry;
    }
    // If there are not enough free pages we need to free some cached pages and allocate into them instead.
    else if(free_pages_shifted_index < header->not_visited_buffer_counter)
    {
        // Get meta information about the page owning the previous allocation.
        const int2 not_visited_memory_page_coords = allocate_pages_push.vsm_not_visited_pages_buffer[free_pages_shifted_index].coords;
        const uint64_t meta_entry = allocate_pages_push.vsm_meta_memory_table.get()[not_visited_memory_page_coords];

        const uint new_vsm_page_entry = pack_meta_coords_to_vsm_entry(not_visited_memory_page_coords) | allocated_mask();

        // Reset previously owning vsm page
        if(get_meta_memory_is_point_light(meta_entry))
        {
            const PointLightCoords owning_vsm_coords = get_vsm_point_light_coords_from_meta_entry(meta_entry);
            const uint point_page_array_index = get_vsm_point_page_array_idx(owning_vsm_coords.face_index, owning_vsm_coords.point_light_index);
            const int3 page_coords = int3(owning_vsm_coords.texel_coords, point_page_array_index);
            allocate_pages_push.vsm_point_page_table[owning_vsm_coords.mip_level].get()[page_coords] = 0u;
        }
        else
        {
            const int3 owning_vsm_coords = get_vsm_coords_from_meta_entry(meta_entry);
            allocate_pages_push.vsm_page_table.get()[owning_vsm_coords] = 0u;
        }

        uint64_t packed_meta_memory_page_entry = 0;
        if(is_point_light)
        {

            const uint point_page_array_index = get_vsm_point_page_array_idx(request.coords.z, request.point_light_index);
            allocate_pages_push.vsm_point_page_table[request.point_light_mip].get()[uint3(request.coords.xy, point_page_array_index)] = new_vsm_page_entry;

            const PointLightCoords vsm_point_light_page_coords = PointLightCoords(
                request.coords.xy,          // texel_coords
                request.point_light_mip,    // mip_level
                request.coords.z,           // face_index
                request.point_light_index); // point_light_index

            packed_meta_memory_page_entry = pack_vsm_point_light_coords_to_meta_entry(vsm_point_light_page_coords) | meta_memory_point_light_mask();
        }
        else 
        {
            allocate_pages_push.vsm_page_table.get()[request.coords] = new_vsm_page_entry;
            allocate_pages_push.vsm_page_view_pos_row.get()[request.coords] = camera_view_last_row;
            packed_meta_memory_page_entry = pack_vsm_coords_to_meta_entry(request.coords);
        }

        // Write the packed coordinates of the virtual vsm page into the meta memory entry.
        const uint64_t alloc_packed_meta_memory_page_entry = packed_meta_memory_page_entry | meta_memory_allocated_mask();
        allocate_pages_push.vsm_meta_memory_table.get()[not_visited_memory_page_coords] = alloc_packed_meta_memory_page_entry;
    }
    // If we have no pages to free (ie we ran out of memory) mark this page as allocation failed.
    else
    {
        allocate_pages_push.vsm_page_table.get()[request.coords] = allocation_failed_mask();
    }
}