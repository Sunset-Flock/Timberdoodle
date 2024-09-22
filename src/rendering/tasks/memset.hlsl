#include "daxa/daxa.inl"

#include "memset.inl"

[[vk::push_constant]] IndirectMemsetBufferPush ind_memset_push;
[shader("compute")]
[numthreads(INDIRECT_MEMSET_BUFFER_X, 1, 1)]
void entry_indmemset(uint3 dtid : SV_DispatchThreadID)
{
    uint index = dtid.x;
    IndirectMemsetBufferCommand command = deref(ind_memset_push.attach.command);
    if (index > command.size)
    {
        return;
    }
    deref_i(ind_memset_push.attach.dst, index + command.offset) = command.value;
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
    uint value = deref_i(memcpy_push.attach.src, index + memcpy_push.src_offset);
    deref_i(memcpy_push.attach.dst, index + memcpy_push.dst_offset) = value;
}