#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "draw_asteroids.inl"
#include "../../shader_shared/asteroids.inl"
#include "../../shader_lib/misc.hlsl"

static const uint HASH_KEY_1 = 15823;
static const uint HASH_KEY_2 = 9737333;
static const uint HASH_KEY_3 = 440817757;

[[vk::push_constant]] RadixDownsweepPassPush radix_count_pass_push;

// We do radix in four passes each sorting 2^8 == 256 bits
groupshared uint count_pass_counters [256];

[numthreads(RADIX_DOWNSWEEP_PASS_WORKGROUP_X, 1, 1)]
[shader("compute")]
void radix_downsweep_pass(
    uint3 svdtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 gtid : SV_GroupThreadID
)
{
    let push = radix_count_pass_push;
    let asteroid_index = svdtid.x;

    count_pass_counters[gtid.x] = 0;
    GroupMemoryBarrierWithGroupSync();

    if(asteroid_index < push.asteroid_count)
    {
        const uint key = push.attach.spatial_hash_src[asteroid_index].x;
        const uint masked_key = (key >> (push.pass_index * 8)) & 0xFF;
        InterlockedAdd(count_pass_counters[masked_key], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Each work group has 256 bins.
    const uint wg_bin_offset = 256 * gid.x;
    push.attach.wg_bin_counts[wg_bin_offset + gtid.x] = count_pass_counters[gtid.x];
}

[[vk::push_constant]] RadixScanPassPush radix_scan_pass_push;

groupshared uint radix_scan_pass_shared_prefix[RADIX_SCAN_PASS_WORKGROUP_X / 32];
groupshared uint per_iteration_prefix;
[numthreads(RADIX_SCAN_PASS_WORKGROUP_X, 1, 1)]
[shader("compute")]
void radix_scan_pass(
    uint3 svdtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 gtid : SV_GroupThreadID
)
{
    let push = radix_scan_pass_push;
    const uint wave_id = gtid.x >> 5;

    for(int iteration_wg_bin_offset = 0; iteration_wg_bin_offset < push.downsweep_pass_wg_count; iteration_wg_bin_offset += RADIX_SCAN_PASS_WORKGROUP_X)
    {
        const bool first_iteration = (iteration_wg_bin_offset == 0);
        // gid specifies the digit for which we are calculating the prefix sum;
        // gtid is the id of the thread within the workgroup - which denotes the bin of each workgroup.
        // Each work group has 256 bins.
        const uint bin_index = gtid.x + iteration_wg_bin_offset;
        const uint wg_bin_offset = 256 * bin_index + gid.x;
        const uint prev_iteration_prefix_result = first_iteration ? 0u : per_iteration_prefix;

        // Load the values of our digit from each workgroup from the global memory into local registers.
        uint count = 0u;
        if(bin_index < push.downsweep_pass_wg_count)
        {
            count = push.attach.wg_bin_counts[wg_bin_offset];
        }

        // Calculate local exclusive sum for each wave.
        const uint wave_exclusive_prefix_count = WavePrefixSum(count);

        // In each wave the last thread writes out the INCLUSIVE sum into shared memory.
        if(WaveGetLaneIndex() == 31)
        {
            radix_scan_pass_shared_prefix[wave_id] = wave_exclusive_prefix_count + count;
        }
        GroupMemoryBarrierWithGroupSync();

        // Only one wave then performs inclusive sum over the values writted for each wave.
        if(wave_id == 0)
        {
            // Load the inclusive sum per wave stored in LDS and calculate the prefix sum.
            const uint per_wave_inclusive_sum = radix_scan_pass_shared_prefix[WaveGetLaneIndex()];
            const uint workgroup_exclusive_sum = WavePrefixSum(per_wave_inclusive_sum);
            radix_scan_pass_shared_prefix[WaveGetLaneIndex()] = workgroup_exclusive_sum;

            if(WaveGetLaneIndex() == 31)
            {
                const uint prev_per_iteration_prefix = first_iteration ? 0u : per_iteration_prefix;
                // Write out the total sum of all iterms in this workgroup - we need the exlusive sum over all results of all waves.
                const uint workgroup_inclusive_sum = workgroup_exclusive_sum + per_wave_inclusive_sum;
                per_iteration_prefix = workgroup_inclusive_sum + prev_per_iteration_prefix;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    
        if(bin_index < push.downsweep_pass_wg_count)
        {
            // Now we have the prefix sum which gives us the sum of all waves that come before us.
            const uint final_prefix_sum = wave_exclusive_prefix_count + radix_scan_pass_shared_prefix[wave_id];
            push.attach.wg_bin_counts[wg_bin_offset] = final_prefix_sum + prev_iteration_prefix_result;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if((wave_id == 0) && WaveIsFirstLane())
    {
        push.attach.wg_bin_counts[push.downsweep_pass_wg_count * 256 + gid.x] = per_iteration_prefix;
        // printf("%d\n", per_iteration_prefix);
    }
}

[[vk::push_constant]] RadixScanFinalizePassPush radix_scan_finalize_pass_push;

groupshared uint gs_radix_scan_finalize_pass[RADIX_SCAN_FINALIZE_PASS_WORKGROUP_X / 32];
[numthreads(RADIX_SCAN_FINALIZE_PASS_WORKGROUP_X, 1, 1)]
[shader("compute")]
void radix_scan_finalize_pass(
    uint3 gtid : SV_GroupThreadID
)
{
    let push = radix_scan_finalize_pass_push;
    const uint wave_id = gtid.x >> 5;
    const uint total_digit_count = push.attach.wg_bin_counts[push.downsweep_pass_wg_count * 256 + gtid.x];

    const uint wave_total_digit_exclusive_prefix_count = WavePrefixSum(total_digit_count);
    if(WaveGetLaneIndex() == 31)
    {
        gs_radix_scan_finalize_pass[wave_id] = wave_total_digit_exclusive_prefix_count + total_digit_count;
    }
    GroupMemoryBarrierWithGroupSync();

    if(wave_id == 0)
    {
        const uint wave_count = RADIX_SCAN_FINALIZE_PASS_WORKGROUP_X / 32;
        if(WaveGetLaneIndex() < wave_count)
        {
            const uint gs_value = gs_radix_scan_finalize_pass[WaveGetLaneIndex()];
            const uint prefix_sum = WavePrefixSum(gs_value);
            gs_radix_scan_finalize_pass[WaveGetLaneIndex()] = prefix_sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint final_sum = wave_total_digit_exclusive_prefix_count + gs_radix_scan_finalize_pass[wave_id];
    push.attach.wg_bin_counts[push.downsweep_pass_wg_count * 256 + gtid.x] = final_sum;
}

[[vk::push_constant]] RadixUpsweepPassPush radix_upsweep_pass_push;

static const uint UINT_BITSIZE = 32;
static const uint DIGIT_COUNT = 256;
static const uint BITMASK_SIZE = DIGIT_COUNT / UINT_BITSIZE;

struct PerDigitBitmask
{
    uint mask[BITMASK_SIZE];
};

groupshared PerDigitBitmask gs_radix_upsweep_bitmask[DIGIT_COUNT];

[numthreads(RADIX_UPSWEEP_PASS_WORKGROUP_X, 1, 1)]
[shader("compute")]
void radix_upsweep_pass(
    uint3 svdtid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 gtid : SV_GroupThreadID
)
{
    let push = radix_upsweep_pass_push;
    let asteroid_index = svdtid.x;

    [[unroll]]
    for(int bitmask_index = 0; bitmask_index < BITMASK_SIZE; ++bitmask_index)
    {
        gs_radix_upsweep_bitmask[gtid.x].mask[bitmask_index] = 0u;
    }

    GroupMemoryBarrierWithGroupSync();

    const uint thread_bitmask_index = gtid.x >> 5;
    const uint thread_bitmask_value = 1u << (gtid.x & 0x1F);

    uint2 key_index;
    uint workgroup_offset;
    uint digit_offset;
    uint digit;
    if(asteroid_index < push.asteroid_count)
    {
        key_index = push.attach.src_spatial_hash[asteroid_index];
        digit = (key_index.x >> (push.pass_index * 8)) & 0xFF;
        workgroup_offset = push.attach.wg_bin_counts[gid.x * 256 + digit];
        digit_offset = push.attach.wg_bin_counts[push.downsweep_pass_wg_count * 256 + digit];
        InterlockedAdd(gs_radix_upsweep_bitmask[digit].mask[thread_bitmask_index], thread_bitmask_value);
    }
    GroupMemoryBarrierWithGroupSync();

    if(asteroid_index < push.asteroid_count)
    {
        uint in_workgroup_offset = 0;
        for(uint bitmask_index = 0; bitmask_index < BITMASK_SIZE; ++bitmask_index)
        {
            uint bits = gs_radix_upsweep_bitmask[digit].mask[bitmask_index];

            uint full_count = countbits(bits);
            uint partial_count = countbits(bits & (thread_bitmask_value - 1));
            in_workgroup_offset += (bitmask_index < thread_bitmask_index) ? full_count : 0u;
            in_workgroup_offset += (bitmask_index == thread_bitmask_index) ? partial_count : 0u;
        }

        push.attach.dst_spatial_hash[digit_offset + workgroup_offset + in_workgroup_offset] = key_index;
    }
}

[[vk::push_constant]] FinalizeHashingPush finalize_hashing_push;

[numthreads(SPATIAL_HASH_FINALIZE_WORKGROUP_X, 1, 1)]
[shader("compute")]
void finalize_hasing(
    uint3 svdtid : SV_DispatchThreadID
)
{
    let push = finalize_hashing_push;
    let asteroid_index = svdtid.x;

    if(asteroid_index < push.asteroid_count)
    {
        const uint current_key = push.attach.spatial_hash[asteroid_index].x;
        const uint prev_key = (asteroid_index == 0) ? uint.maxValue : push.attach.spatial_hash[asteroid_index - 1].x;

        if(current_key != prev_key)
        {
            push.attach.cell_start_indices[current_key] = asteroid_index;
        }
    }
}