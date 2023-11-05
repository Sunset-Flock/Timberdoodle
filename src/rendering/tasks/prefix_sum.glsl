#include <daxa/daxa.inl>

#include "prefix_sum.inl"

#if defined(PrefixSumWriteCommand_COMMAND)
DAXA_DECL_PUSH_CONSTANT(PrefixSumWriteCommandPush, push)
layout(local_size_x = 1) in;
void main()
{
    const uint value_count = deref(u_value_count[push.uint_offset]);
    const uint block_count = (value_count + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
    DispatchIndirectValueCount command_and_count;
    
    command_and_count.command.x = block_count;
    command_and_count.command.y = 1;
    command_and_count.command.z = 1;
    command_and_count.value_count = value_count;
    deref(u_upsweep_command0) = command_and_count;
     
    command_and_count.command.x = (block_count + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
    command_and_count.command.y = 1;
    command_and_count.command.z = 1;
    command_and_count.value_count = block_count;
    deref(u_upsweep_command1) = command_and_count;

    command_and_count.command.x = block_count;
    command_and_count.command.y = 1;
    command_and_count.command.z = 1;
    command_and_count.value_count = value_count;
    deref(u_downsweep_command) = command_and_count;
}
#endif

#if defined(UPSWEEP)
#define SHARED_PREFIX_SUM_VALUE_COUNT ( PREFIX_SUM_BLOCK_SIZE / WARP_SIZE )
shared uint shared_prefix_sum_values[SHARED_PREFIX_SUM_VALUE_COUNT];
void prefix_sum(
    uint warp_index,
    uint warp_id,
    inout uint value)
{
    value = subgroupInclusiveAdd(value);
    if (warp_index == (WARP_SIZE - 1))
    {
        shared_prefix_sum_values[warp_id] = value;
    }
    memoryBarrierShared();
    barrier();
    // Only one warp of size 32 is required to prefix sum the shared values, as 32 * 32 = 1024
    if (warp_id == 0)
    {
        uint shared_value = shared_prefix_sum_values[warp_index];
        uint shared_sum = subgroupInclusiveAdd(shared_value);
        shared_prefix_sum_values[warp_index] = shared_sum;
    }
    memoryBarrierShared();
    barrier();
    value += shared_prefix_sum_values[warp_id - 1];
}
DAXA_DECL_PUSH_CONSTANT(PrefixSumPush, push)
layout(local_size_x = PREFIX_SUM_BLOCK_SIZE) in;
void main()
{
    const uint global_index = gl_GlobalInvocationID.x;
    const uint warp_id = gl_SubgroupID;
    const uint warp_index = gl_SubgroupInvocationID;
    const uint src_index = global_index * push.uint_src_stride + push.uint_src_offset;
    const uint value_count = deref(u_command).value_count;

    uint value = 0;
    if (global_index < value_count)
    {
        value = deref(u_src[src_index]);
    }
    prefix_sum(
        warp_index,
        warp_id,
        value);
    if (gl_LocalInvocationID .x == (PREFIX_SUM_BLOCK_SIZE - 1)) 
    {
        deref(u_block_sums[gl_WorkGroupID.x]) = value;
    }
    if (global_index < value_count)
    {
        const uint out_index = global_index * push.uint_dst_stride + push.uint_dst_offset;
        deref(u_dst[out_index]) = value;
    }
}
#endif // #if defined(UPSWEEP)

#if defined(DOWNSWEEP)
DAXA_DECL_PUSH_CONSTANT(PrefixSumPush, push)
layout(local_size_x = PREFIX_SUM_BLOCK_SIZE) in;
void main()
{
    const uint value_index = gl_GlobalInvocationID.x + PREFIX_SUM_BLOCK_SIZE; // Skip the first block.
    const uint left_block_index = gl_WorkGroupID.x;
    const uint value_count = deref(u_command).value_count;

    if (value_index >= value_count)
    {
        return;
    }

    const uint block_sum = deref(u_block_sums[left_block_index]);
    const uint index = push.uint_dst_offset + push.uint_dst_stride * value_index;
    deref(u_values[index]) = deref(u_values[index]) + block_sum;
}
#endif // #if defined(DOWNSWEEP)