#include "daxa/daxa.inl"

#include "memset.inl"

[[vk::push_constant]] IndirectMemsetBufferPush ind_memset_push;
[shader("compute")]
[numthreads(INDIRECT_MEMSET_BUFFER_X, 1, 1)]
void entry_indmemset(uint3 dtid : SV_DispatchThreadID)
{
    uint index = dtid.x;
    IndirectMemsetBufferCommand command = *ind_memset_push.attachments.command;
    if (index > command.size)
    {
        return;
    }
    ind_memset_push.attachments.dst[index + command.offset] = command.value;
}

[[vk::push_constant]] MemcpyBufferPush memcpy_push;
[shader("compute")]
[numthreads(MEMCPY_BUFFER_X, 1, 1)]
void entry_memcpy(uint3 dtid : SV_DispatchThreadID)
{
    uint index = dtid.x;
    if (index >= memcpy_push.size)
    {
        return;
    }
    uint value = memcpy_push.attachments.src[index + memcpy_push.src_offset];
    memcpy_push.attachments.dst[index + memcpy_push.dst_offset] = value;
}