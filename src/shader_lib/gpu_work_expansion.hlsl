#pragma once

#include <daxa/daxa.inl>

#include "misc.hlsl"
#include "../shader_shared/gpu_work_expansion.inl"

struct ExpandedWorkItem
{
    uint src_work_item_index;
    uint dst_work_item_index;
};
interface WorkExpansionGetDstWorkItemCountI
{
    func get_itemcount(uint src_work_item_index) -> uint;
};

/// WARNING: Cant be member function due to a slang compiler bug!
func po2_expansion_get_workitem<SrcWrkItmT : WorkExpansionGetDstWorkItemCountI>(Po2WorkExpansionBufferHead * self, SrcWrkItmT src_work_items, uint thread_index, uint bucket_index, out ExpandedWorkItem ret) -> bool
{
    let argument_index = thread_index >> bucket_index;
    let argument_count = self.bucket_sizes[bucket_index];
    if (argument_index >= argument_count)
    {
        return false;
    }
    let in_argument_work_offset = thread_index - (argument_index << bucket_index);
    let src_work_item_index = (self.buckets[bucket_index])[argument_index];
    let work_item_count = src_work_items.get_itemcount(src_work_item_index);
    // Now we need to find the offset of our work.
    // The offset for each argument is simply the sum of all the dst work items from all the smaller arg buckets for this src work item.
    // To get the that sum we simply take the work item count and mask off all bits from the buckets bit upwards.
    let smaller_buckets_mask = ~((~0u) << bucket_index);
    let arg_dst_work_offset = work_item_count & smaller_buckets_mask;
    let dst_work_item_index = in_argument_work_offset + arg_dst_work_offset;
    ret.src_work_item_index = src_work_item_index;
    ret.dst_work_item_index = dst_work_item_index;
    return true;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func po2_expansion_add_workitems(Po2WorkExpansionBufferHead * self, uint dst_item_count, uint src_work_item_index, uint dst_workgroup_size_log2)
{
    let orig_dst_item_count = dst_item_count;
    let dst_workgroup_size = 1u << dst_workgroup_size_log2;
    while(dst_item_count != 0)
    {
        let bit_index = firstbithigh(dst_item_count);

        let bucket_index = bit_index;
        dst_item_count = dst_item_count & ~(1u << bit_index);

        uint bucket_count_prev_value = 0;
        InterlockedAdd(self.bucket_sizes[bucket_index], 1, bucket_count_prev_value);
        let bucket_arg_index = bucket_count_prev_value;
        let bucket_capacity = capacity_of_bucket(self.max_src_items, self.max_dst_items, bucket_index);
        let arg_allocation_success = bucket_arg_index < bucket_capacity;
        if (arg_allocation_success)
        {
            (self.buckets[bucket_index])[bucket_arg_index] = src_work_item_index;

            // Now we also need to update the indirect argument
            let total_work_args = bucket_arg_index + 1;
            let total_threads_needed = total_work_args << bucket_index;
            let total_workgroups_needed = ((total_threads_needed + dst_workgroup_size - 1) >> dst_workgroup_size_log2);
            uint dummy;
            InterlockedMax(self.bucket_dispatches[bucket_index].x, total_workgroups_needed, dummy);
        }
        else
        {
            // TODO: WRITE NUMBER OF FAILED ALLOCS TO READBACK BUFFER!
            printf("ALLOC FAILED bucket_arg_index %i, bucket_capacity %i\n", bucket_arg_index, bucket_capacity);
        }
    }
}

/// WARNING: Cant be member function due to a slang compiler bug!
func prefix_sum_expansion_get_workitem<SrcWrkItmT : WorkExpansionGetDstWorkItemCountI>(PrefixSumExpansionBufferHead * self, SrcWrkItmT src_work_items, uint thread_index, out ExpandedWorkItem ret) -> bool
{
    // Each thread needs a dst work item.
    // To get a dst work item a thread needs to be assigned to a DstWorkItemGroup and then pick a dst work item out the group.
    // Threads are simply linearly allocated to all DstWorkItemGroups.

    // Finding what dst work item group a thread belongs to is easy.
    // Each thread belongs to the first DistWorkItemGroup that has a larger prefix sum value than the threads index.
    // To make the search faster we use a binary search instead of a linear search.
    
    daxa::u32 count = min(daxa::u32(self->merged_src_item_dst_item_count), self->dwig_capacity);
    daxa::u32 window_end = count; // Exclusive end is one past the last element.
    daxa::u32 window_begin = 0;

    // Search until there is only one possible element
    while ((window_end - window_begin) > 1) {
        const daxa::u32 window_size = window_end - window_begin;
        const daxa::u32 cursor = window_begin + ((window_size - 1) / 2); // We want the cursor to be smaller index biased.
        const daxa::u32 prefix_sum_val = self->dwig_inclusive_dst_work_item_count_prefix_sum[cursor];

        // When the prefix sum value is greater than the thread index, 
        // It could potentially be the correct array index we search for.
        // So we include this value for the next search iteration window.
        // But as we only care about THE FIRST one that is larger, we cut off every other element past it.
        const bool greater = prefix_sum_val > thread_index;
        window_end = select(greater, cursor + 1, window_end);

        // When the prefix sum value is smaller or equal to the threads index, 
        // it can not be the right dst work item group.
        // This is because we need the fist dst work item group that has a LARGER prefix sum value .
        const bool less_equal = !greater;
        window_begin = select(less_equal, cursor + 1, window_begin);
    }

    if (window_begin >= count || window_begin <= window_end)
    {
        printf("ERROR BINARY SEARCH FAILED\n");
        return false;
    }

    const uint dwig_index = window_begin;
    const uint dwig_prefix_sum_value = self->dwig_inclusive_dst_work_item_count_prefix_sum[dwig_index];
    const bool found_index = dwig_prefix_sum_value > thread_index;
    if (!found_index)
    {
        ret = (ExpandedWorkItem)0;
        return false;
    }

    const uint src_work_item_index = self->dwig_src_work_items[dwig_index];
    const uint dwig_dst_item_count = src_work_items.get_itemcount(src_work_item_index);
    const uint dst_work_item_index = thread_index - (dwig_prefix_sum_value - dwig_dst_item_count);

    // This case is possible under normal circumstances for the last few threads that may overhang for the last dwig.
    const bool dst_work_item_valid = dst_work_item_index < dwig_dst_item_count;
    if (!dst_work_item_valid)
    {
        ret = (ExpandedWorkItem)0;
        return false;
    }

    ret.src_work_item_index = src_work_item_index;
    ret.dst_work_item_index = dst_work_item_index;
    return true;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func prefix_sum_expansion_add_workitems(PrefixSumExpansionBufferHead* self, uint dst_item_count, uint src_item_index, uint dst_workgroup_size_log2)
{
    // Single merged atomic 64 bit atomic performing two 32 bit atomic adds for prefix sum and dwig append.
    daxa::u64 prev_merged_count = AtomicAddU64(self->merged_src_item_dst_item_count, daxa::u64(1) << 32 | daxa::u64(dst_item_count));
    daxa::u32 out_dst_work_item_group = uint(prev_merged_count >> 32) - 1;
    daxa::u32 inclusive_dst_item_count_prefix_sum = uint(prev_merged_count);
    if (out_dst_work_item_group < self->dwig_capacity)
    {
        self->dwig_inclusive_dst_work_item_count_prefix_sum[out_dst_work_item_group] = inclusive_dst_item_count_prefix_sum;
        self->dwig_src_work_items[out_dst_work_item_group] = src_item_index;
        // Its ok if we launch a few too many threads.
        // The get_dst_workitem function is robust against that.
        daxa::u32 needed_workgroups = round_up_div_btsft(inclusive_dst_item_count_prefix_sum, dst_workgroup_size_log2);
        InterlockedMax(self->dispatch.x, needed_workgroups);
    }
}