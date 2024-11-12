#pragma once

#include <daxa/daxa.inl>

#include "misc.hlsl"
#include "../shader_shared/gpu_work_expansion.inl"

#define PO2_COMPACT 0

struct DstItemInfo
{
    uint src_item_index;
    uint in_expansion_index;
};

/// WARNING: FUNCTION EXPECTS ALL THREADS IN THE WARP TO BE ACTIVE.
/// WARNING: Cant be member function due to a slang compiler bug!
func po2packed_expansion_get_workitem(Po2PackedWorkExpansionBufferHead * self, uint thread_index, out DstItemInfo ret) -> bool
{
    // This code ensures that each wave only has a single bucket to work on. It is never the case that some threads within the warp work on different buckets.
    // This is very important, as it allows warps to cooperatively search for the bucket they should work on.
    // Each thread checks, if the wave should work on one of the 32 buckets.
    // The thread that finds the waves first thread index in one of the buckets ranges gets elected and broadcasts the bucket the wave will work on.
    let lanes_bucket_index = WaveGetLaneIndex();
    let lanes_bucket_threads = self->bucket_thread_counts[lanes_bucket_index];
    let lanes_bucket_threads_rounded_to_wave_multiple = round_up_to_multiple_po2(lanes_bucket_threads, WARP_SIZE);      // All threads within a warp must work on the same bucket.
    let lanes_bucket_first_thread = WavePrefixSum(lanes_bucket_threads_rounded_to_wave_multiple);        
    let lanes_bucket_one_past_last_thread = lanes_bucket_first_thread + lanes_bucket_threads_rounded_to_wave_multiple;

    let warp_first_thread = thread_index & (~WARP_SIZE_MULTIPLE_MASK);
    let warp_belongs_to_lanes_bucket = warp_first_thread >= lanes_bucket_first_thread && warp_first_thread < lanes_bucket_one_past_last_thread;

    let bucket_index_ballot = WaveActiveBallot(warp_belongs_to_lanes_bucket).x;
    if (bucket_index_ballot == 0)
    {
        // Overhang elimination
        ret = (DstItemInfo)0;
        return false;
    }

    let bucket_index = firstbitlow(bucket_index_ballot);
    let bucket_first_thread_index = WaveShuffle(lanes_bucket_first_thread, bucket_index);
    let bucket_threads = WaveShuffle(lanes_bucket_threads, bucket_index);
    let bucket_arg_count = round_up_div_btsft(bucket_threads, bucket_index);
    let bucket_relative_thread_index = thread_index - bucket_first_thread_index;

    // We MUST check the current arg AND the next arc.
    // This is because the args only mark the first thread within each arg.
    // So we might have threads here that belong to the next arg, as those are only marked in the next arg.
    let first_arg_idx = round_down_div_btsft(bucket_relative_thread_index, bucket_index);
    let secnd_arg_idx = min(first_arg_idx + 1, bucket_arg_count - 1);
    let first_expansion_idx = self->buckets[bucket_index][first_arg_idx];
    let secnd_expansion_idx = self->buckets[bucket_index][secnd_arg_idx];
    let secnd_expansion_src_item_index = self->expansions[secnd_expansion_idx].src_item_index;

    // Distribute field loads across wave. Improves perf a tiny but on nvidia.
    let fast_load_start_addr = (uint*)(self->expansions + first_expansion_idx);
    uint lane_v = 0;
    if (WaveGetLaneIndex() < 3)
    {
        lane_v = fast_load_start_addr[WaveGetLaneIndex()];
    }
    uint first_expansion_src_item_index = WaveBroadcastLaneAt(lane_v, 0);
    uint first_expansion_factor = WaveBroadcastLaneAt(lane_v, 1);
    uint first_expansion_first_thread_in_bucket = WaveBroadcastLaneAt(lane_v, 2);

    if (bucket_relative_thread_index >= bucket_threads)
    {
        // Overhang elimination
        ret = (DstItemInfo)0;
        return false;
    }
    
    let in_first_expansion = bucket_relative_thread_index < (first_expansion_first_thread_in_bucket + first_expansion_factor);
    ret.src_item_index = select(in_first_expansion, first_expansion_src_item_index, secnd_expansion_src_item_index);
    ret.in_expansion_index = bucket_relative_thread_index - select(in_first_expansion, first_expansion_first_thread_in_bucket, first_expansion_first_thread_in_bucket + first_expansion_factor);
    return true;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func po2packed_expansion_add_workitems(Po2PackedWorkExpansionBufferHead * self, uint expansion_factor, uint src_item_index, uint dst_workgroup_size_log2)
{
    let orig_dst_item_count = expansion_factor;
    let dst_workgroup_size = 1u << dst_workgroup_size_log2;

    uint expansion_index = 0;
    InterlockedAdd(self->expansions_count, 1, expansion_index);
    const uint cur_expansion_count = expansion_index + 1;

    if (expansion_index >= self->expansions_max)
    {
        // TODO: WRITE NUMBER OF FAILED ALLOCS TO READBACK BUFFER!
        printf("GPU ERROR: work expansion failed, ran out of memory, expansion. %i, max expansions: %i\n", expansion_index, self->expansions_max);
        return;
    }

    // Update total dst threads needed.
    uint prev_dst_item_count = 0;
    InterlockedAdd(self->expansions_thread_count, expansion_factor, prev_dst_item_count);
    const uint cur_dst_item_count = prev_dst_item_count + expansion_factor;

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
    // The expansions dont need to be exactly 2^bucket sized.
    // This means that the threads for one arg can overlap into the following arg.
    // This is ok, as we store the first thread in bucket in the expansion.
    // Dst threads read the previous expansion arg and determine which of the two expansions they belong to.
    // This allows us to have all threads for a given expansion to be in the same arg always.
    // This imporves cache locality for meshlet culling AND drawing as well as reduce atomic ops in expansion append in mesh culling.
    // It is also easier to debug.
    uint bucket = firstbithigh(expansion_factor);

    uint first_thread_in_bucket = 0;
    InterlockedAdd(self->bucket_thread_counts[bucket], expansion_factor, first_thread_in_bucket);
    uint last_thread_in_bucket = first_thread_in_bucket + expansion_factor - 1;

    if (last_thread_in_bucket >= WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS)
    {
        // TODO: WRITE NUMBER OF FAILED ALLOCS TO READBACK BUFFER!
        printf("GPU ERROR: work expansion failed, attempted to add threads %i to %i, max expansion threads: %i\n", first_thread_in_bucket, last_thread_in_bucket, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS);
        return;
    }

    self->expansions[expansion_index] = WorkExpansion(src_item_index, expansion_factor, first_thread_in_bucket);

    // We mark each arg in which the first thread is part of the current expansion.
    const uint first_arg = round_up_div_btsft(first_thread_in_bucket, bucket);
    const uint last_arg = round_down_div_btsft(last_thread_in_bucket, bucket);

    self->buckets[bucket][first_arg] = expansion_index;
    if (first_arg != last_arg)
    {
        self->buckets[bucket][last_arg] = expansion_index;
    }
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
    if (thread_index >= total_work_item_count)
    {
        ret = (DstItemInfo)0;
        return false;
    }

    daxa::u32 expansion_count = min(daxa::u32(merged_counters >> 32), self->expansions_max);
    daxa::u32 window_end = expansion_count; // Exclusive end is one past the last element.
    daxa::u32 window_begin = 0;

    // Search until there is only one possible element
    while ((window_end - window_begin) > 1) {
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

    if (window_begin >= expansion_count || window_begin >= window_end)
    {
        // Overhang threads.
        ret = (DstItemInfo)0;
        return false;
    }

    const uint expansion_index = window_begin;
    const uint expansion_inc_prefix_value = self->expansions_inclusive_prefix_sum[expansion_index];
    const bool found_expansion = expansion_index < expansion_count;
    if (!found_expansion)
    {
        printf("GPU ERROR: did not find expansion index for thread: %i, expansion_index: %i, expansion count: %i\n", thread_index, expansion_index, expansion_count);
        ret = (DstItemInfo)0;
        return false;
    }

    const uint src_item_index = self->expansions_src_work_item[expansion_index];
    const uint expansion_factor = self->expansions_expansion_factor[expansion_index];
    const uint in_expansion_index = thread_index - (expansion_inc_prefix_value - expansion_factor);

    // This case is possible under normal circumstances for the last few threads that may overhang for the last expansion.
    const bool dst_work_item_valid = in_expansion_index < expansion_factor;
    if (!dst_work_item_valid)
    {
        printf("GPU ERROR: dst item invalid: %i, dst item count: %i, thread: %i, expansion index index %i, expansion count: %i\n", in_expansion_index, expansion_factor, thread_index, expansion_index, expansion_count);
        ret = (DstItemInfo)0;
        return false;
    }

    ret.src_item_index = src_item_index;
    ret.in_expansion_index = in_expansion_index;
    return true;
}

/// WARNING: Cant be member function due to a slang compiler bug!
func prefix_sum_expansion_add_workitems(PrefixSumWorkExpansionBufferHead* self, uint expansion_factor, uint src_item_index, uint dst_workgroup_size_log2)
{
    // Single merged atomic 64 bit atomic performing two 32 bit atomic adds for prefix sum and expansion append.
    daxa::u64 prev_merged_count = AtomicAddU64(self->merged_expansion_count_thread_count, daxa::u64(1) << 32 | daxa::u64(expansion_factor));
    daxa::u32 expansion_index = uint(prev_merged_count >> 32);
    daxa::u32 inclusive_dst_item_count_prefix_sum = uint(prev_merged_count) + expansion_factor;
    if (expansion_index < self->expansions_max)
    {
        self->expansions_inclusive_prefix_sum[expansion_index] = inclusive_dst_item_count_prefix_sum;
        self->expansions_src_work_item[expansion_index] = src_item_index;
        self->expansions_expansion_factor[expansion_index] = expansion_factor;
        // Its ok if we launch a few too many threads.
        // The get_dst_workitem function is robust against that.
        daxa::u32 needed_workgroups = round_up_div_btsft(inclusive_dst_item_count_prefix_sum, dst_workgroup_size_log2);
        InterlockedMax(self->dispatch.x, needed_workgroups);
    }
}