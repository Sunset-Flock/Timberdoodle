#pragma once

#include <daxa/daxa.inl>

#include "../shader_shared/po2_expansion.inl"

struct Po2ExpandedWorkItem
{
    uint src_work_item_index;
    uint dst_work_item_index;
};
interface IPo2SrcWorkItems
{
    func get_itemcount(uint src_work_item_index) -> uint;
};

// Cant be member function because of slang bugs.
func get_expanded_work_item<SrcWrkItmT : IPo2SrcWorkItems>(Po2WorkExpansionBufferHead * self, SrcWrkItmT src_work_items, uint thread_index, uint bucket_index, out Po2ExpandedWorkItem ret) -> bool
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

// Cant be member function because of slang bugs.
func expand_work_items(Po2WorkExpansionBufferHead * self, uint dst_item_count, uint src_work_item_index, uint dst_workgroup_size_log2)
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