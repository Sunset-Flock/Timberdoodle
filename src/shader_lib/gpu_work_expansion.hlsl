#pragma once

#include <daxa/daxa.inl>

#include "misc.hlsl"
#include "debug.glsl"
#include "../shader_shared/gpu_work_expansion.inl"

#define PO2_COMPACT 0

struct DstItemInfo
{
    uint2 payload;
    uint work_item_index;
};

func po2bucket_get_arg(Po2BucketWorkExpansionBufferHead * self, uint bucket, uint arg) -> uint*
{
    // Pairs of buckets (even and odd index) share a stack section for their args.
    // The even one grows upwards and the odd one downwards into the same memory section.

    // Slang pointer arithmetic is broken for int offsets, cast to int64_t to avoid this bug.
    uint* ret;
    let is_bucket_downwards = (bucket & 0x1) != 0;
    let offset = (is_bucket_downwards ? -int64_t(arg) : int64_t(arg)) * sizeof(uint);
    return (uint*)(int64_t(self->bucket_arg_array_ptrs[bucket]) + offset);
}

/// WARNING: FUNCTION EXPECTS ALL THREADS IN THE WARP TO BE ACTIVE.
/// WARNING: Cant be member function due to a slang compiler bug!
func po2bucket_expansion_get_workitem(Po2BucketWorkExpansionBufferHead * self, uint thread_index, out DstItemInfo ret) -> bool
{
    // It is always the case that ALL threads within the SAME warp work on the SAME bucket.
    // This is very important, as it allows warps to cooperatively search for the bucket they should work on.
    // We have to search for the bucket to work on, because all threads live within a single dispatch, we have to partition all threads in the dispatch into the buckets.
    // We map each thread of the 32 threads to one of the 32 buckets and check if the warp belongs in that bucket.
    // The prefix sum over 'threads per bucket' is not computed before, so we perform it here with a warp accelerated prefix sum.
    let lanes_bucket_index = WaveGetLaneIndex();
    let lanes_bucket_threads = self->bucket_thread_counts[lanes_bucket_index];
    let lanes_bucket_threads_rounded_to_wave_multiple = round_up_to_multiple_po2(lanes_bucket_threads, WARP_SIZE);      // All threads within a warp must work on the same bucket.
    let lanes_bucket_first_thread = WavePrefixSum(lanes_bucket_threads_rounded_to_wave_multiple);        
    let lanes_bucket_one_past_last_thread = lanes_bucket_first_thread + lanes_bucket_threads_rounded_to_wave_multiple;

    let warp_first_thread = thread_index & (~WARP_SIZE_MULTIPLE_MASK);
    let warp_belongs_to_lanes_bucket = warp_first_thread >= lanes_bucket_first_thread && warp_first_thread < lanes_bucket_one_past_last_thread;

    // Uniform over warp.
    let bucket_index_ballot = WaveActiveBallot(warp_belongs_to_lanes_bucket).x;
    if (bucket_index_ballot == 0)
    {
        // Overhang elimination
        ret = (DstItemInfo)0;
        return false;
    }

    let bucket_index = firstbitlow(bucket_index_ballot);
    let bucket_first_thread_index = WaveReadLaneAt(lanes_bucket_first_thread, bucket_index);
    let bucket_threads = WaveReadLaneAt(lanes_bucket_threads, bucket_index);
    let bucket_arg_count = round_up_div_btsft(bucket_threads, bucket_index);
    let bucket_relative_thread_index = thread_index - bucket_first_thread_index;

    // Divergent over warp.
    if (bucket_relative_thread_index >= bucket_threads)
    {
        // Overhang elimination
        ret = (DstItemInfo)0;
        return false;
    }

    // Each expansion within a bucket has between 2^i and 2^(i+1)-1 work items.
    // Ascending thread indices are assigned tightly to all work items of the expansions.
    // Example:
    //   Bucket Index (i):            2
    //   Bucket Expansion Sizes:      2^i = 2^2 = 4 <-> 2^(i+1)-1 = 2^(2+1)-1 = 7
    //
    //   Expansion Index:             0 1 2 3
    //   Expansion Work Item Counts:  7 4 6 4
    //   NOTE: all work item counts are between 4 (2^2) and 7 (2^(2+1)-1)!
    // 
    //   Thread Index:                0  1  2  3  4  5  6    7  8  9 10   11 12 13 14 15 16   17 18 19 20
    //   Threads Expansions:          0  0  0  0  0  0  0    1  1  1  1    2  2  2  2  2  2    3  3  3  3 
    //   Threads Expansion Work item: 0  1  2  3  4  5  6    0  1  2  3    0  1  2  3  4  5    0  1  2  3
    //
    // How does each work item thread know what to work on (how do they know their work item index and their expansion index?)?
    //   * a uint pair per work item would be the simplest solution
    //   * write a pair of uints (work item idx + expansion idx) per work item
    //   * work item threads could use their thread index to index the array of (work item idx, expansion idx) uint pairs
    //   * writing and storing a uint pair per work item is too expensive in runtime and memory.
    //
    // Simple improvement over the uint pair solution:
    //   * we store the first work item thread idx INSIDE the expansion
    //   * allows us to remove the work item index from the uint pair
    //   * work item threads can calculate their work item index via: work_item_idx = thread_idx - expansion.first_thread_idx
    //   * already saves us half the required memory, now we only store an array of expansion indices
    //
    // The limited work item count of each expansion within each bucket (2^i = 2^2 = 4 <-> 2^(i+1)-1 = 2^(2+1)-1 = 7), gives us several guarantees.
    // If we view a range of threads in the bucket of size 2^i = 4 anywhere within the bucket:
    //   * there can be AT MOST two different expansions the threads within the range work on
    //     * this is because the minimum work item count of each expansion is 2^i = 4;
    //       to have more than 2 expansions in a range of 4 threads, we would need expansions with less than 3 work items
    //   * when there is more than one expansion in the range, the second expansion MUST also extend into the next range of 2^i = 4 threads
    //     * this is because the minimum work item count of each expansion is 2^i = 4;
    //       the second expansion would have to be 3 or less work items in order to not reach into the next range of 2^i = 4 threads
    //   * when there is two expansions within a range there MUST BE a range next to the current range of at least 1 thread
    //     * this is because of the previous guarantee
    //
    // Using the first guarantee we know that we only have to store two expansion indices per range!
    // This is already a big improvement compared to storing and writing a expansion index for every thread.
    // For larger bucket sizes, like i = 12 this means we save 4094 of 4096 writes and memory, thats a saving of over 99%.
    // 
    // The other two guarantees give us another big advantage: we only need to store ONE of the two possible expansion indices within each range!
    // For example, if we only store the first expansion index for each range:
    //   * for each range we can load that expansion using the (thread_idx % (2^i = 4)) as index
    //     * load that expansions first thread index
    //     * check if the thread belongs to that expansion (thread_idx < expansion_first_thread_idx + expansion_work_item_count)
    //   * in the case that we have two expansions in the range, some threads will fail this check as they belong to the second expansion of the range.
    //     * it is GUARANTEED that the next range's first expansion also has threads in our range (see above point 3 in the guarantees)!
    //     * as each range stores their first expansion, we can load the first expansion of the next range, all remaining threads must fall into this expansion!
    // 
    // Here is an example of the argument array storing each ranges second expansion index:
    //
    //   Expansion Index:             0  1  2  3
    //   Expansion First Thread:      0  7 11 17
    //   Expansion Work Item Counts:  7  4  6  4
    //
    //   Thread Index:                0  1  2  3    4  5  6  7    8  9 10 11   12 13 14 15   16 17 18 19   20
    //   Threads Expansions:          0  0  0  0    0  0  0  1    1  1  1  2    2  2  2  2    2  3  3  3    3 
    //   Threads Expansion Work item: 0  1  2  3    4  5  6  0    1  2  3  0    1  2  3  4    5  0  1  2    3
    //
    //   Range Arguments:             0             0             1             2             2             3
    //
    // Every work item thread now has to load the argument at (thread-index % 4) AND the argument before that (if present).
    // From both arguments we load the expansion indices
    // From both expansion indices we load the expansions
    // From both expansions we load the first thread indices and work item counts
    // Finally we test into which expansion the thread falls into
    // And the work item index the thread has to work on for its expansion
    let first_arg_idx = round_down_div_btsft(bucket_relative_thread_index, bucket_index);
    let secnd_arg_idx = min(first_arg_idx + 1, bucket_arg_count - 1);
    let first_expansion_idx = *po2bucket_get_arg(self, bucket_index, first_arg_idx);
    let second_expansion_idx = *po2bucket_get_arg(self, bucket_index, secnd_arg_idx);
    let first_expansion = self->expansions[first_expansion_idx];
    let first_expansion_payload = first_expansion.payload;
    let first_expansion_factor = first_expansion.work_item_count;
    let first_expansion_first_thread_in_bucket = first_expansion.first_thread_in_bucket;
    let secnd_expansion_payload = self->expansions[second_expansion_idx].payload;
    
    let in_first_expansion = bucket_relative_thread_index < (first_expansion_first_thread_in_bucket + first_expansion_factor);
    ret.payload = select(in_first_expansion, first_expansion_payload, secnd_expansion_payload);
    ret.work_item_index = bucket_relative_thread_index - select(in_first_expansion, first_expansion_first_thread_in_bucket, first_expansion_first_thread_in_bucket + first_expansion_factor);
    return true;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func po2bucket_expansion_add_workitems(Po2BucketWorkExpansionBufferHead * self, uint work_item_count, uint2 payload, uint dst_workgroup_size_log2)
{
    let orig_dst_item_count = work_item_count;
    let dst_workgroup_size = 1u << dst_workgroup_size_log2;

    uint expansion_index = 0;
    InterlockedAdd(self->expansion_count, 1, expansion_index);
    const uint cur_expansion_count = expansion_index + 1;

    GPU_ASSERT_COMPARE_INT(expansion_index, <, self->expansions_max);
    if (expansion_index >= self->expansions_max)
    {
        return;
    }

    // Update total dst threads needed.
    uint prev_dst_item_count = 0;
    InterlockedAdd(self->expansions_thread_count, work_item_count, prev_dst_item_count);
    const uint cur_dst_item_count = prev_dst_item_count + work_item_count;

    // Update indirect dispatch:
    {
        // We MUST dispatch "too many" threads.
        // This is because argument bucket sizes are rounded up in the dst threads to ensure each warp only has a single argument bucket.
        // Because of this there are a few threads between buckets lost. 
        // The lost threads must be made up by conservatively adding some overhang warps (32 is the save number here).
        // As the workgroup sizes determine the dispatch we also calculate the needed extra workgroups to get 32 extra warps.
        daxa::u32 extra_workgroups = div_btsft(32 * WARP_SIZE, dst_workgroup_size_log2);
        daxa::u32 needed_workgroups = round_up_div_btsft(cur_dst_item_count, dst_workgroup_size_log2) + extra_workgroups;
        InterlockedMax(self->dispatch.x, needed_workgroups);
    }

    // Select the bucket with the most fitting expansion ratio.
    // Expansions with N >= 2^bucket && N < 2^(bucket+1) go into the same bucket.
    // The expansions don't need to be exactly 2^bucket sized.
    // This means that the threads for one arg can overlap into the following arg.
    // This is ok, as we store the first thread in bucket in the expansion.
    // Dst threads read the previous expansion arg and determine which of the two expansions they belong to.
    // This allows us to have all threads for a given expansion to be in the same arg always.
    // This improves cache locality for meshlet culling AND drawing as well as reduce atomic ops in expansion append in mesh culling.
    // It is also easier to debug.
    uint bucket = firstbithigh(work_item_count);

    uint first_thread_in_bucket = 0;
    InterlockedAdd(self->bucket_thread_counts[bucket], work_item_count, first_thread_in_bucket);
    uint last_thread_in_bucket = first_thread_in_bucket + work_item_count - 1;

    GPU_ASSERT_COMPARE_INT(last_thread_in_bucket, <, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS);
    if (last_thread_in_bucket >= WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS)
    {
        return;
    }

    self->expansions[expansion_index] = Po2BucketWorkExpansion(payload, work_item_count, first_thread_in_bucket);

    // We mark each arg in which the first thread is part of the current expansion.
    const uint first_arg = round_up_div_btsft(first_thread_in_bucket, bucket);
    const uint last_arg = round_down_div_btsft(last_thread_in_bucket, bucket);

    *po2bucket_get_arg(self, bucket, first_arg) = expansion_index;
    if (first_arg != last_arg)
    {
        *po2bucket_get_arg(self, bucket, last_arg) = expansion_index;
    }
}

func subone_log32_floor(uint v) -> uint
{
    let v_sub_one = max(1,v)-1;
    let v_log2_floor = firstbithigh(v_sub_one);
    let log2_32 = 5;
    let v_log32_ceil = v_log2_floor / log2_32;
    return v_log32_ceil;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func prefix_sum_expansion_get_workitem(PrefixSumWorkExpansionBufferHead * self, uint thread_index, out DstItemInfo ret) -> bool
{
    // Each thread needs a dst work item.
    // To get a dst work item a thread needs to be assigned to a DstWorkItemGroup and then pick a dst work item out the group.
    // Threads are simply linearly allocated to all DstWorkItemGroups.

    // Finding what dst work item group a thread belongs to is easy.
    // Each thread belongs to the first DistWorkItemGroup that has a larger prefix sum value than the threads index.
    // To make the search faster we use a binary search instead of a linear search.
    daxa::u64 merged_counters = self->merged_expansion_count_thread_count;
    daxa::u32 total_work_item_count = daxa::u32(merged_counters);

    daxa::u32 expansion_count = min(daxa::u32(merged_counters >> 32), self->expansions_max);
    daxa::u32 window_end = expansion_count; // Exclusive end is one past the last element.
    daxa::u32 window_begin = 0;
    #if 0
    let warp_first_thread = thread_index & (~WARP_SIZE_MULTIPLE_MASK);

    let iterations = 3;//subone_log32_floor(expansion_count);
    uint lane_section_size = 32 << (5 * (3)); // calculates pow(32, iterations)
    for (uint i = 0; i < 3; ++i)
    {
        lane_section_size = lane_section_size / 32;
        let window_size = window_end - window_begin;
        let lane_cursor = min(window_end, window_begin + lane_section_size * (WaveGetLaneIndex() + 1)) - 1;
        let lane_prefix_value = self->expansions_inclusive_prefix_sum[lane_cursor];
        let greater = lane_prefix_value > warp_first_thread;
        let greater_ballot = WaveActiveBallot(greater).x;
        if (greater_ballot == 0)
        {
            window_begin = window_end = expansion_count;
            break;
        }
        let first_lane_greater = firstbitlow(greater_ballot);
        window_begin = window_begin + first_lane_greater * lane_section_size;
        window_end = min(window_end, window_begin + lane_section_size);
    }


    let warp_window_begin = window_begin;
    let lane_prefix_v = self->expansions_inclusive_prefix_sum[min(window_begin + WaveGetLaneIndex(), window_end-1)];

    // Search until there is only one possible element
    for (uint i = 0; i < 5; ++i)
    {
        const daxa::u32 window_size = window_end - window_begin;
        const daxa::u32 cursor = window_begin + ((window_size - 1) / 2); // We want the cursor to be smaller index biased.
        //const daxa::u32 prefix_sum_val = self->expansions_inclusive_prefix_sum[cursor];
        const daxa::u32 prefix_sum_val = WaveShuffle(lane_prefix_v, cursor - warp_window_begin);

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
    #else
    // Search until there is only one possible element
    while ((window_end - window_begin) > 1)
    {
        const daxa::u32 window_size = window_end - window_begin;
        const daxa::u32 cursor = window_begin + ((window_size - 1) / 2); // We want the cursor to be smaller index biased.
        const daxa::u32 prefix_sum_val = self->expansions_inclusive_prefix_sum[cursor];

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
    #endif
    if (thread_index >= total_work_item_count)
    {
        ret = (DstItemInfo)0;
        return false;
    }

    if (window_begin >= expansion_count || window_begin >= window_end)
    {
        // Overhang threads.
        ret = (DstItemInfo)0;
        return false;
    }

    const uint expansion_index = window_begin;
    const uint expansion_inc_prefix_value = self->expansions_inclusive_prefix_sum[expansion_index];
    const bool found_expansion = expansion_index < expansion_count;
    GPU_ASSERT(found_expansion)
    if (!found_expansion)
    {
        ret = (DstItemInfo)0;
        return false;
    }

    const uint2 payload = self->expansions_payloads[expansion_index];
    const uint expansion_factor = self->expansions_expansion_factor[expansion_index];
    const uint in_expansion_index = thread_index - (expansion_inc_prefix_value - expansion_factor);

    // This case is possible under normal circumstances for the last few threads that may overhang for the last expansion.
    const bool dst_work_item_valid = in_expansion_index < expansion_factor;
    GPU_ASSERT(found_expansion)
    if (!dst_work_item_valid)
    {
        ret = (DstItemInfo)0;
        return false;
    }

    ret.payload = payload;
    ret.work_item_index = in_expansion_index;
    return true;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func prefix_sum_expansion_add_workitems(PrefixSumWorkExpansionBufferHead* self, uint work_items, uint2 payload, uint dst_workgroup_size_log2)
{
    // Single merged atomic 64 bit atomic performing two 32 bit atomic adds for prefix sum and expansion append.
    daxa::u64 prev_merged_count = AtomicAddU64(self->merged_expansion_count_thread_count, daxa::u64(1) << 32 | daxa::u64(work_items));
    daxa::u32 expansion_index = uint(prev_merged_count >> 32);
    daxa::u32 inclusive_dst_item_count_prefix_sum = uint(prev_merged_count) + work_items;
    if (expansion_index < self->expansions_max)
    {
        self->expansions_inclusive_prefix_sum[expansion_index] = inclusive_dst_item_count_prefix_sum;
        self->expansions_payloads[expansion_index] = payload;
        self->expansions_expansion_factor[expansion_index] = work_items;
        // Its ok if we launch a few too many threads.
        // The get_dst_workitem function is robust against that.
        daxa::u32 needed_workgroups = round_up_div_btsft(inclusive_dst_item_count_prefix_sum, dst_workgroup_size_log2);
        InterlockedMax(self->dispatch.x, needed_workgroups);
    }
}