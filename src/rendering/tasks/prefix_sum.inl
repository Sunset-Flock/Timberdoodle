#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/asset.inl"

#define PREFIX_SUM_BLOCK_SIZE 1024
#define PREFIX_SUM_WORKGROUP_SIZE PREFIX_SUM_BLOCK_SIZE

struct DispatchIndirectValueCount
{
    DispatchIndirectStruct command;
    daxa_u32 value_count;
};
DAXA_DECL_BUFFER_PTR(DispatchIndirectValueCount)

#if __cplusplus || defined(PrefixSumWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(PrefixSumWriteCommand, 1)
DAXA_TASK_USE_BUFFER(u_value_count, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_upsweep_command0, daxa_RWBufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_WRITE)
DAXA_TASK_USE_BUFFER(u_upsweep_command1, daxa_RWBufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_WRITE)
DAXA_TASK_USE_BUFFER(u_downsweep_command, daxa_RWBufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || defined(UPSWEEP)
DAXA_DECL_TASK_USES_BEGIN(PrefixSumUpsweep, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_src, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_dst, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_TASK_USE_BUFFER(u_block_sums, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || defined(DOWNSWEEP)
DAXA_DECL_TASK_USES_BEGIN(PrefixSumDownsweep, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_block_sums, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_values, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

struct PrefixSumWriteCommandPush
{
    daxa_u32 uint_offset;
};

struct PrefixSumPush
{
    daxa_u32 uint_src_offset;
    daxa_u32 uint_src_stride;
    daxa_u32 uint_dst_offset;
    daxa_u32 uint_dst_stride;
};

#if __cplusplus

#include "../gpu_context.hpp"
#include "misc.hpp"

static constexpr inline char const PREFIX_SUM_SHADER_PATH[] = "./src/rendering/tasks/prefix_sum.glsl";

using PrefixSumCommandWriteTask = WriteIndirectDispatchArgsPushBaseTask<
    PrefixSumWriteCommand,
    PREFIX_SUM_SHADER_PATH,
    PrefixSumWriteCommandPush
>;

// Sums n <= 1024 values up. Always writes 1024 values out (for simplicity in multi pass).
struct PrefixSumUpsweepTask
{
    DAXA_USE_TASK_HEADER(PrefixSumUpsweep)
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
            .compile_options = {
                .defines = {{"UPSWEEP", "1"}},
            },
        },
        .push_constant_size = sizeof(PrefixSumPush),
        .name = std::string{PrefixSumUpsweep::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumPush push = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(PrefixSumUpsweep::NAME));
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

struct PrefixSumDownsweepTask
{
    DAXA_USE_TASK_HEADER(PrefixSumDownsweep)
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
            .compile_options = {
                .defines = {{"DOWNSWEEP", "1"}},
            },
        },
        .push_constant_size = sizeof(PrefixSumPush),
        .name = std::string{PrefixSumDownsweep::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumPush push = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(PrefixSumDownsweep::NAME));
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

// Task function that can prefix sum up to 2^20 million values.
// Reads values from src buffer with src_offset and src_stride,
// writes values to dst buffer with dst_offset and dst_stride.
struct PrefixSumTaskGroupInfo
{
    GPUContext * context;
    daxa::TaskGraph& task_list;
    u32 max_value_count;
    u32 value_count_uint_offset;
    daxa::TaskBufferView value_count_buf;
    u32 src_uint_offset;
    u32 src_uint_stride;
    daxa::TaskBufferView src_buf;
    u32 dst_uint_offset;
    u32 dst_uint_stride;
    daxa::TaskBufferView dst_buf;
};
void task_prefix_sum(PrefixSumTaskGroupInfo info)
{
    DAXA_DBG_ASSERT_TRUE_M(info.max_value_count < (1 << 20), "max max value is 2^20");
    auto upsweep0_command_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "prefix sum upsweep0_command_buffer",
    });
    auto upsweep1_command_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "prefix sum upsweep1_command_buffer",
    });
    auto downsweep_command_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "prefix sum downsweep_command_buffer",
    });
    info.task_list.add_task(PrefixSumCommandWriteTask{
        .uses={
            .u_value_count = info.value_count_buf,
            .u_upsweep_command0 = upsweep0_command_buffer,
            .u_upsweep_command1 = upsweep1_command_buffer,
            .u_downsweep_command = downsweep_command_buffer,
        },
        .context = info.context,
        .push = {info.value_count_uint_offset}
    });
    auto max_block_count = (static_cast<u64>(info.max_value_count) + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
    auto block_sums_src = info.task_list.create_transient_buffer({
        .size = static_cast<u32>(sizeof(u32) * max_block_count),
        .name = "prefix sum block_sums_src",
    });
    info.task_list.add_task(PrefixSumUpsweepTask{
        .uses={
            .u_command = upsweep0_command_buffer,
            .u_src = info.src_buf,
            .u_dst = info.dst_buf,
            .u_block_sums = block_sums_src,
        },
        .context = info.context,
        .push = {
            .uint_src_offset = info.src_uint_offset,
            .uint_src_stride = info.src_uint_stride,
            .uint_dst_offset = info.dst_uint_offset,
            .uint_dst_stride = info.dst_uint_stride,
        },
    });
    auto block_sums_dst = info.task_list.create_transient_buffer({
        .size = static_cast<u32>(sizeof(u32) * max_block_count),
        .name = "prefix sum block_sums_dst",
    });
    auto total_count = info.task_list.create_transient_buffer({
        .size = static_cast<u32>(sizeof(u32)),
        .name = "prefix sum block_sums total count",
    });
    info.task_list.add_task(PrefixSumUpsweepTask{
        .uses={
            .u_command = upsweep1_command_buffer,
            .u_src = block_sums_src,
            .u_dst = block_sums_dst,
            .u_block_sums = total_count,
        },
        .context = info.context,
        .push = {
            .uint_src_offset = 0,
            .uint_src_stride = 1,
            .uint_dst_offset = 0,
            .uint_dst_stride = 1,
        },
    });
    info.task_list.add_task(PrefixSumDownsweepTask{
        .uses={
            .u_command = downsweep_command_buffer,
            .u_block_sums = block_sums_dst,
            .u_values = info.dst_buf,
        },
        .context = info.context,
        .push = {
            .uint_src_offset = std::numeric_limits<u32>::max(),
            .uint_src_stride = std::numeric_limits<u32>::max(),
            .uint_dst_offset = info.dst_uint_offset,
            .uint_dst_stride = info.dst_uint_stride,
        },
    });
}

#endif