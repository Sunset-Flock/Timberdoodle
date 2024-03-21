#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"

#define INDIRECT_MEMSET_BUFFER_X 128
#define MEMCPY_BUFFER_X 128

struct IndirectMemsetBufferCommand
{
    DispatchIndirectStruct dispatch;
    // In u32's
    daxa_u32 offset;
    daxa_u32 size;
    daxa_u32 value;
};
DAXA_DECL_BUFFER_PTR(IndirectMemsetBufferCommand)

DAXA_DECL_TASK_HEAD_BEGIN(IndirectMemsetBufferH, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IndirectMemsetBufferCommand), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), dst)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(MemcpyBufferH, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(daxa_u32), src)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), dst)
DAXA_DECL_TASK_HEAD_END

struct IndirectMemsetBufferPush
{
    DAXA_TH_BLOB(IndirectMemsetBufferH, uses)
    daxa_u32 dummy;
};

struct MemcpyBufferPush
{
    DAXA_TH_BLOB(MemcpyBufferH, uses)
    // In u32's
    daxa_u32 src_offset;
    daxa_u32 dst_offset;
    daxa_u32 size;
};

#if defined(__cplusplus)

using IndirectMemsetBufferTask = SimpleIndirectComputeTask<
    IndirectMemsetBufferH::Task, 
    IndirectMemsetBufferPush, 
    "./src/rendering/tasks/memset.slang",
    "entry_indmemset"
>;
using MemcpyBufferTask = SimpleComputeTask<
    MemcpyBufferH::Task, 
    MemcpyBufferPush, 
    "./src/rendering/tasks/memset.slang", 
    "entry_memcpy"
>;

#endif // #if defined(__cplusplus)