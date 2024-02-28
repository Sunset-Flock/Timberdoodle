#include <daxa/daxa.inl>

#include "memset.inl"

#if defined(IndirectMemsetBuffer_SHADER)
DAXA_DECL_PUSH_CONSTANT(IndirectMemsetBufferPush, push)
layout(local_size_x = INDIRECT_MEMSET_BUFFER_X) in;
void main()
{
    uint index = gl_GlobalInvocationID.x;
    IndirectMemsetBufferCommand command = deref(push.attachments.command);
    if (index > command.size)
    {
        return;
    }
    deref(push.attachments.dst[index + command.offset]) = command.value;
}
#endif // #if defined(IndirectMemsetBuffer_SHADER)

#if defined(MemcpyBuffer_SHADER)
DAXA_DECL_PUSH_CONSTANT(MemcpyBufferPush, push)
layout(local_size_x = MEMCPY_BUFFER_X) in;
void main()
{
    uint index = gl_GlobalInvocationID.x;
    if (index >= push.size)
    {
        return;
    }
    uint value = deref(push.attachments.src[index + push.src_offset]);
    deref(push.attachments.dst[index + push.dst_offset]) = value;
}
#endif // #if defined(MemcpyBuffer_SHADER)