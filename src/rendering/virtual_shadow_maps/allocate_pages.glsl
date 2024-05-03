#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

#extension GL_EXT_debug_printf : enable

DAXA_DECL_PUSH_CONSTANT(AllocatePagesH, push)
layout (local_size_x = ALLOCATE_PAGES_X_DISPATCH) in;
void main()
{
    // - Read the values stored inside FindFreePages header
    //   - If the GlobalThreadID is less than free_buffer_counter:
    //     - Read the entry in FreePageBuffer[GlobalThreadID]
    //     - Read the entry in AllocationRequest[GlobalThreadID]
    //     - Assign new entries to the page_table_texel and meta_memory_texel
    //   - If the (GlobalThreadID - free_buffer_counter) < not_visited_buffer_counter:
    //     - Read the entry in NotVisitedPageBuffer[GlobalThreadID]
    //     - Read the meta memory entry
    //     - Reset (Deallocate) the entry that previously owned this memory in virtual page table 
    //     - Assign new entries to the page_table_texel and meta_memory_texel
    FindFreePagesHeader header = deref(push.vsm_find_free_pages_header);

    const int id = int((gl_GlobalInvocationID.z * ALLOCATE_PAGES_X_DISPATCH) + gl_LocalInvocationID.x);
    if(id >= deref(push.vsm_allocation_count).count) { return; }

    const int free_shifted_id = id - int(header.free_buffer_counter);

    const ivec3 alloc_request_page_coords = deref_i(push.vsm_allocation_requests, id).coords;
    const bool allocated = deref_i(push.vsm_allocation_requests, id).already_allocated != 0;

    const int current_camera_height_offset = deref_i(push.vsm_clip_projections, alloc_request_page_coords.z).height_offset;
    if(!allocated)
    {
        // Use up all free pages
        if(id < header.free_buffer_counter)
        {
            // debugPrintfEXT("allocating from id %d\n", id);
            const ivec2 free_memory_page_coords = deref_i(push.vsm_free_pages_buffer, id).coords;
        
            uint new_vsm_page_entry = pack_meta_coords_to_vsm_entry(free_memory_page_coords);
            new_vsm_page_entry |= allocated_mask();
            imageStore(daxa_uimage2DArray(push.vsm_page_table), alloc_request_page_coords, uvec4(new_vsm_page_entry));
            imageStore(daxa_iimage2DArray(push.vsm_page_height_offsets), alloc_request_page_coords, ivec4(current_camera_height_offset));

            uint new_meta_memory_page_entry = pack_vsm_coords_to_meta_entry(alloc_request_page_coords);
            new_meta_memory_page_entry |= meta_memory_allocated_mask();
            imageStore(daxa_uimage2D(push.vsm_meta_memory_table), free_memory_page_coords, uvec4(new_meta_memory_page_entry));
        } 
        // If there is not enough free pages free NOT VISITED pages to make space
        else if (free_shifted_id < header.not_visited_buffer_counter)
        {
            const ivec2 not_visited_memory_page_coords = deref_i(push.vsm_not_visited_pages_buffer, free_shifted_id).coords;

            // Reset previously owning vsm page
            const uint meta_entry = imageLoad(daxa_uimage2D(push.vsm_meta_memory_table), not_visited_memory_page_coords).r;
            const ivec3 owning_vsm_coords = get_vsm_coords_from_meta_entry(meta_entry);
            imageStore(daxa_uimage2DArray(push.vsm_page_table), owning_vsm_coords, uvec4(0));
        
            // Perform the allocation
            uint new_vsm_page_entry = pack_meta_coords_to_vsm_entry(not_visited_memory_page_coords);
            new_vsm_page_entry |= allocated_mask();
            imageStore(daxa_uimage2DArray(push.vsm_page_table), alloc_request_page_coords, uvec4(new_vsm_page_entry));
            imageStore(daxa_iimage2DArray(push.vsm_page_height_offsets), alloc_request_page_coords, ivec4(current_camera_height_offset));

            uint new_meta_memory_page_entry = pack_vsm_coords_to_meta_entry(alloc_request_page_coords);
            new_meta_memory_page_entry |= meta_memory_allocated_mask();
            imageStore(daxa_uimage2D(push.vsm_meta_memory_table), not_visited_memory_page_coords, uvec4(new_meta_memory_page_entry));
        } 
        // Else mark the page as allocation failed
        else 
        {
            imageStore(daxa_uimage2DArray(push.vsm_page_table), alloc_request_page_coords, uvec4(allocation_failed_mask()));
        }
    } 
    else
    {
        imageStore(daxa_iimage2DArray(push.vsm_page_height_offsets), alloc_request_page_coords, ivec4(current_camera_height_offset));
    }
}
    