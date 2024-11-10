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

/// WARNING: FUNCTION EXPECTS ALL THREADS IN THE WARP TO BE ACTIVE.
/// WARNING: Cant be member function due to a slang compiler bug!
func po2_expansion_get_workitem<SrcWrkItmT : WorkExpansionGetDstWorkItemCountI>(Po2WorkExpansionBufferHead * self, SrcWrkItmT src_work_items, uint thread_index, out ExpandedWorkItem ret) -> bool
{
    // This code ensures that each wave only has a single bucket to work on. It is never the case that some threads within the warp work on different buckets.
    // This is very important, as it allows warps to cooperatively search for the bucket they should work on.
    // Each thread checks, if the wave should work on one of the 32 buckets.
    // The thread that finds the waves first thread index in one of the buckets ranges gets elected and broadcasts the bucket the wave will work on.
    let lanes_bucket_index = WaveGetLaneIndex();
    let lanes_bucket_size = self->bucket_sizes[lanes_bucket_index];
    let lanes_bucket_threads = lanes_bucket_size << lanes_bucket_index;                        // bucket size counds how many args are in the bucket. Nr. of threads is arg_count<<bucket_index.
    let lanes_bucket_threads_rounded_to_wave_multiple = round_up_to_multiple(lanes_bucket_threads, WARP_SIZE);      // All threads within a warp must work on the same bucket.
    let lanes_bucket_first_thread = WavePrefixSum(lanes_bucket_threads_rounded_to_wave_multiple);        
    let lanes_bucket_one_past_last_thread = lanes_bucket_first_thread + lanes_bucket_threads_rounded_to_wave_multiple;

    let warp_first_thread = thread_index & (~WARP_SIZE_MULTIPLE_MASK);
    let warp_belongs_to_lanes_bucket = warp_first_thread >= lanes_bucket_first_thread && warp_first_thread < lanes_bucket_one_past_last_thread;

    if (!WaveActiveAnyTrue(warp_belongs_to_lanes_bucket))
    {
        // Overhang elimination
        ret = (ExpandedWorkItem)0;
        return false;
    }

    let bucket_index = firstbitlow(WaveActiveBallot(warp_belongs_to_lanes_bucket).x);
    let bucket_first_thread_index = WaveShuffle(lanes_bucket_first_thread, bucket_index);
    let bucket_argument_count = WaveShuffle(lanes_bucket_size, bucket_index);
    let bucket_relative_thread_index = thread_index - bucket_first_thread_index;
    let bucket_argument_index = bucket_relative_thread_index >> bucket_index;
    let bucket_in_argument_index = bucket_relative_thread_index - (bucket_argument_index << bucket_index);

    if (bucket_argument_index >= bucket_argument_count)
    {    
        // Overhang elimination
        ret = (ExpandedWorkItem)0;
        return false;
    }

    let src_work_item_index = (self.buckets[bucket_index])[bucket_argument_index];
    let src_work_item_expansion_factor = src_work_items.get_itemcount(src_work_item_index);
    // Now we need to find the offset of our work.
    // The offset for each argument is simply the sum of all the dst work items from all the smaller arg buckets for this src work item.
    // To get the that sum we simply take the work item count and mask off all bits from the buckets bit upwards.
    let src_work_item_smaller_buckets_mask = ~((~0u) << bucket_index);
    let src_work_item_offset = src_work_item_expansion_factor & src_work_item_smaller_buckets_mask;

    let dst_work_item_index = bucket_in_argument_index + src_work_item_offset;

    if (dst_work_item_index >= src_work_item_expansion_factor)
    {
        // Overhang elimination
        ret = (ExpandedWorkItem)0;
        return false;
    }

    ret.src_work_item_index = src_work_item_index;
    ret.dst_work_item_index = dst_work_item_index;
    return true;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func po2_expansion_add_workitems(Po2WorkExpansionBufferHead * self, uint dst_item_count, uint src_work_item_index, uint dst_workgroup_size_log2)
{
    let orig_dst_item_count = dst_item_count;
    let dst_workgroup_size = 1u << dst_workgroup_size_log2;

    uint prev_src_item_count = 0;
    InterlockedAdd(self->src_work_item_count, 1, prev_src_item_count);
    const uint cur_src_item_count = prev_src_item_count + 1;

    if (cur_src_item_count >= self->buckets_capacity)
    {
        // TODO: WRITE NUMBER OF FAILED ALLOCS TO READBACK BUFFER!
        printf("ERROR: work expansion failed, ran out of memory, src item nr. %i, capacity: %i\n", cur_src_item_count, self->buckets_capacity);
    }

    // Update total dst threads needed.
    uint prev_dst_item_count = 0;
    InterlockedAdd(self->dst_work_item_count, dst_item_count, prev_dst_item_count);
    const uint cur_dst_item_count = prev_dst_item_count + dst_item_count;

    // Update indirect dispatch:
    {
        // We MUST dispatch "too many" threads.
        // This is because argument bucket sizes are rounded up in the dst threads to ensure each warp only has a single argument bucket.
        // Because of this there are a few threads between buckets lost. 
        // The lost threads must be made up by conservatively adding some overhang warps (32 is the save number here).
        // As the workgroup sizes determine the dispatch we also calculate the needed extra workgroups to get 32 extra warps.
        daxa::u32 extra_workgroups = div_btsft(32*WARP_SIZE, dst_workgroup_size_log2);
        daxa::u32 needed_workgroups = round_up_div_btsft(cur_dst_item_count, dst_workgroup_size_log2) + extra_workgroups;
        InterlockedMax(self->dispatch.x, needed_workgroups);
    }
    
    while(dst_item_count != 0)
    {
        let bit_index = firstbithigh(dst_item_count);

        let bucket_index = bit_index;
        dst_item_count = dst_item_count & ~(1u << bit_index);

        uint bucket_count_prev_value = 0;
        InterlockedAdd(self.bucket_sizes[bucket_index], 1, bucket_count_prev_value);
        let bucket_arg_index = bucket_count_prev_value;
        let arg_allocation_success = bucket_arg_index < self->buckets_capacity;

        if (arg_allocation_success)
        {
            (self.buckets[bucket_index])[bucket_arg_index] = src_work_item_index;
        }
        else
        {
            printf("ERROR: CRITICAL LOGIC ERROR, ran out of memory when inserting into bucket, bucket idx: %i, bucket arg idx: %i, capacity: %i\n", bucket_index, bucket_arg_index, self->buckets_capacity);
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
    daxa::u64 merged_counters = self->merged_src_item_dst_item_count;
    daxa::u32 total_work_item_count = daxa::u32(merged_counters);
    if (thread_index >= total_work_item_count)
    {
        ret = (ExpandedWorkItem)0;
        return false;
    }

    daxa::u32 dwig_count = min(daxa::u32(merged_counters >> 32), self->dwig_capacity);
    daxa::u32 window_end = dwig_count; // Exclusive end is one past the last element.
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

    if (window_begin >= dwig_count || window_begin >= window_end)
    {
        // Overhang threads.
        ret = (ExpandedWorkItem)0;
        return false;
    }

    const uint dwig_index = window_begin;
    const uint dwig_prefix_sum_value = self->dwig_inclusive_dst_work_item_count_prefix_sum[dwig_index];
    const bool found_dwig = dwig_index < dwig_count;
    if (!found_dwig)
    {
        printf("ERROR: did not find dwig for thread: %i, dwig_index: %i, dwig count: %i\n", thread_index, dwig_index, dwig_count);
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
        printf("ERROR: dst item invalid: %i, dst item count: %i, thread: %i, dwig index %i, dwig count: %i\n", dst_work_item_index, dwig_dst_item_count, thread_index, dwig_index, dwig_count);
        ret = (ExpandedWorkItem)0;
        return false;
    }

    ret.src_work_item_index = src_work_item_index;
    ret.dst_work_item_index = dst_work_item_index;
    return true;
}

#define WARP_BINARY_SEARCH_SPEEDUP 1

/// WARNING: FUNCTION EXPECTS ALL THREADS IN THE WARP TO BE ACTIVE.
/// WARNING: Cant be member function due to a slang compiler bug!
func cooperative_prefix_sum_expansion_get_workitem<SrcWrkItmT : WorkExpansionGetDstWorkItemCountI>(PrefixSumExpansionBufferHead * self, SrcWrkItmT src_work_items, uint thread_index, out ExpandedWorkItem ret) -> bool
{
    daxa::u64 merged_counters = self->merged_src_item_dst_item_count;
    daxa::u32 total_work_item_count = daxa::u32(merged_counters);
    daxa::u32 dwig_count = min(daxa::u32(merged_counters >> 32), self->dwig_capacity);

    /// WARNING: ALL THREADS MUST BE ACTIVE HERE AS WE USE WARP INTRINSICS!!!!
    // if (thread_index >= total_work_item_count)
    // {
    //     ret = (ExpandedWorkItem)0;
    //     return false;
    // }

    daxa::u32 window_end = dwig_count; // Exclusive end is one past the last element.
    daxa::u32 window_begin = 0;

    let warp_first_thread = thread_index & (~WARP_SIZE_MULTIPLE_MASK);
    let lane_index = WaveGetLaneIndex();

    uint iter = 0;
    while ((window_end - window_begin) >= 32)
    {
        uint window_size = window_end - window_begin;
        // Segment remaining dwig into 32 ranges.
        // Each thread processes one of these ranges.
        let lanes_window_begin = window_begin + (window_size * lane_index) / WARP_SIZE;
        let lanes_window_end = min(window_begin + window_size * (lane_index + 1) / WARP_SIZE, dwig_count);
        // Each thread performs a SINGLE LOAD per iteration. 
        let lanes_window_begin_threads = self->dwig_exclusive_dst_work_item_count_prefix_sum[lanes_window_begin];
        // Read right lanes result to get own right bound.
        daxa::u32 lanes_right_lane = min(lane_index + 1, WARP_SIZE - 1);
        daxa::u32 lanes_window_end_threads2_partial = WaveShuffle(lanes_window_begin_threads, lanes_right_lane);
        daxa::u32 lanes_window_end_threads = (lane_index == (WARP_SIZE - 1)) ? total_work_item_count : lanes_window_end_threads2_partial;        // As the last thread has no right thread to read from it must get its value manually.
        bool warp_first_thread_in_lane_window = warp_first_thread >= lanes_window_begin_threads && warp_first_thread < lanes_window_end_threads;
        if (!WaveActiveAnyTrue(warp_first_thread_in_lane_window))
        {
            // printf("lanes_window_first_dwig_threads %i, lanes_window_begin_threads %i, lanes_window_end_threads %i\n", lanes_window_first_dwig_threads, lanes_window_begin_threads, lanes_window_end_threads);
            ret = (ExpandedWorkItem)0;
            return false;
        }
        let first_thread_window_lane = firstbitlow(WaveActiveBallot(warp_first_thread_in_lane_window).x);
        window_begin = window_begin + (window_size * first_thread_window_lane) / WARP_SIZE;
        window_end = min(window_begin + window_size * (first_thread_window_lane + 1) / WARP_SIZE, dwig_count);
        iter += 1;
        //if (iter > 3)
        //    break;
    }

    // Refine search for other threads in warp.
    // Must increase search radius for all threads with index > 0 as we only searched for the first threads index.
    window_end = min(window_begin + 32, dwig_count);

    // Need to save the window begin when threads read their values in as the window begin changes within the search.
    let initial_window_begin = window_begin;
    let lanes_prefix_value_index = min(window_begin + WaveGetLaneIndex(), dwig_count - 1);
    let lanes_prefix_value = self->dwig_inclusive_dst_work_item_count_prefix_sum[lanes_prefix_value_index];

    // Search until there is only one possible element
    while (WaveActiveAnyTrue((window_end - window_begin) > 1)) {
        const daxa::u32 window_size = window_end - window_begin;
        const daxa::u32 cursor = window_begin + ((window_size - 1) / 2); // We want the cursor to be smaller index biased.
        const daxa::u32 prefix_sum_val = WaveShuffle(lanes_prefix_value, cursor - initial_window_begin);

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

    if (window_begin >= dwig_count || window_begin >= window_end)
    {
        printf("ERROR: binary search failed in work expansion\n");
        ret = (ExpandedWorkItem)0;
        return false;
    }

    const uint dwig_index = window_begin;
    const uint dwig_prefix_sum_value = self->dwig_inclusive_dst_work_item_count_prefix_sum[dwig_index];
    const bool found_dwig = dwig_index < dwig_count;
    if (!found_dwig)
    {
        printf("ERROR: did not find dwig for thread: %i, dwig_index: %i, dwig count: %i\n", thread_index, dwig_index, dwig_count);
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
        // Its legal and intended for threads to enter here.
        // This is because we dispatch a multiple of 32 or 128 threads.
        // The overhang threads are killed here.
        // printf("ERROR: dst item invalid: %i, dst item count: %i, thread: %i, dwig index %i, dwig count: %i\n", dst_work_item_index, dwig_dst_item_count, thread_index, dwig_index, dwig_count);
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
    daxa::u32 out_dst_work_item_group = uint(prev_merged_count >> 32);
    daxa::u32 inclusive_dst_item_count_prefix_sum = uint(prev_merged_count) + dst_item_count;
    if (out_dst_work_item_group < self->dwig_capacity)
    {
        self->dwig_inclusive_dst_work_item_count_prefix_sum[out_dst_work_item_group] = inclusive_dst_item_count_prefix_sum;
        self->dwig_exclusive_dst_work_item_count_prefix_sum[out_dst_work_item_group] = uint(prev_merged_count);
        self->dwig_src_work_items[out_dst_work_item_group] = src_item_index;
        // Its ok if we launch a few too many threads.
        // The get_dst_workitem function is robust against that.
        daxa::u32 needed_workgroups = round_up_div_btsft(inclusive_dst_item_count_prefix_sum, dst_workgroup_size_log2) + (1 << (dst_workgroup_size_log2));
        InterlockedMax(self->dispatch.x, needed_workgroups);
    }
}