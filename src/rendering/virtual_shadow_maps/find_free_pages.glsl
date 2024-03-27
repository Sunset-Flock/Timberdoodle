#include <daxa/daxa.inl>
#include "shader_lib/vsm_util.glsl"
#include "shader_shared/vsm_shared.inl"
#include "vsm.inl"

struct BufferReserveInfo
{
    uint reserved_count;
    uint reserved_offset;
    uint order;
    bool condition;
};

BufferReserveInfo count_pages_and_reserve_buffer_slots(
    bool count_free_pages,
    daxa_BufferPtr(FindFreePagesHeader) header,
    uint meta_entry
)
{
    bool condition;
    if(count_free_pages) { condition = !get_meta_memory_is_allocated(meta_entry); }
    else                 { condition =  get_meta_memory_is_allocated(meta_entry) && (!get_meta_memory_is_visited(meta_entry)); }

    const uvec4 condition_mask = subgroupBallot(condition);
    const uint order = subgroupBallotExclusiveBitCount(condition_mask);

    uint broadcast_value = 0;
    // Last thread will attempt to allocate all the pages
    if(gl_SubgroupInvocationID == 31)
    {
        // Because we did ExlusiveSum earlier we also need to recheck this threads result
        // as it was not included in the sum
        const uint page_count = order + int(condition);

        uint previous_counter_value;
        if(count_free_pages) { previous_counter_value = atomicAdd(deref(header).free_buffer_counter, page_count); } 
        else                 { previous_counter_value = atomicAdd(deref(header).not_visited_buffer_counter, page_count); }

        uint reserve_count = 0;
        uint counter_overflow = 0;
        if(previous_counter_value < MAX_VSM_ALLOC_REQUESTS)
        {
            const uint counter_capacity = MAX_VSM_ALLOC_REQUESTS - previous_counter_value;
            reserve_count = min(page_count, counter_capacity);
            counter_overflow = page_count - reserve_count;
        } else {
            counter_overflow = page_count;
        }

        // fix the counter if it overflowed
        if(count_free_pages) { atomicAdd(deref(header).free_buffer_counter, -counter_overflow); }
        else                 { atomicAdd(deref(header).not_visited_buffer_counter, -counter_overflow); }

        // Pack reserve data into a single uint so we can use a single broadcast to distribute it
        // MSB 16 - reserved offset
        // LSB 16 - reserved count
        broadcast_value |= previous_counter_value << 16;
        broadcast_value |= (reserve_count & n_mask(16));
    }

    uint reserved_info = subgroupBroadcast(broadcast_value, 31);
    return BufferReserveInfo(
        reserved_info & n_mask(16), // reserved_count
        reserved_info >> 16,        // reserved_offset
        order,                      // thread order
        condition                   // condition
    );
}

DAXA_DECL_PUSH_CONSTANT(FindFreePagesH, push)
layout (local_size_x = FIND_FREE_PAGES_X_DISPATCH, local_size_y = 1, local_size_z = 1) in;
void main()
{
    if(gl_GlobalInvocationID.x == 0)
    {
        const uint allocations_number = deref(push.vsm_allocation_count).count;

        const uint allocate_dispach_count = (allocations_number + ALLOCATE_PAGES_X_DISPATCH - 1) / ALLOCATE_PAGES_X_DISPATCH;
        deref(push.vsm_allocate_indirect).x = 1;
        deref(push.vsm_allocate_indirect).y = 1;
        deref(push.vsm_allocate_indirect).z = allocate_dispach_count;

        deref(push.vsm_clear_indirect).x = VSM_PAGE_SIZE / CLEAR_PAGES_X_DISPATCH;
        deref(push.vsm_clear_indirect).y = VSM_PAGE_SIZE / CLEAR_PAGES_Y_DISPATCH;
        deref(push.vsm_clear_indirect).z = deref(push.vsm_allocation_count).count;

        // const daxa_u32 clear_dirty_bit_distpach_count = 
        //     (allocations_number + VSM_CLEAR_DIRTY_BIT_LOCAL_SIZE_X - 1) / VSM_CLEAR_DIRTY_BIT_LOCAL_SIZE_X;
        // deref(_vsm_clear_dirty_bit_indirect).x = 1;
        // deref(_vsm_clear_dirty_bit_indirect).y = 1;
        // deref(_vsm_clear_dirty_bit_indirect).z = clear_dirty_bit_distpach_count;
    }

    const int linear_thread_index = int(gl_WorkGroupID.x * FIND_FREE_PAGES_X_DISPATCH + gl_LocalInvocationID.x);
    const ivec2 thread_coords = ivec2(
        linear_thread_index % VSM_META_MEMORY_TABLE_RESOLUTION,
        linear_thread_index / VSM_META_MEMORY_TABLE_RESOLUTION
    );

    const uint meta_entry = imageLoad(daxa_uimage2D(push.vsm_meta_memory_table), thread_coords).r;

    BufferReserveInfo info = count_pages_and_reserve_buffer_slots(true, push.vsm_find_free_pages_header, meta_entry);
    bool fits_into_reserved_slots = info.order < info.reserved_count;
    if(info.condition && fits_into_reserved_slots) 
    {
        deref_i(push.vsm_free_pages_buffer, info.reserved_offset + info.order).coords = thread_coords;
    }

    info = count_pages_and_reserve_buffer_slots(false, push.vsm_find_free_pages_header, meta_entry);
    fits_into_reserved_slots = info.order < info.reserved_count;
    if(info.condition && fits_into_reserved_slots) 
    {
        deref_i(push.vsm_not_visited_pages_buffer,info.reserved_offset + info.order).coords = thread_coords;
    }
}