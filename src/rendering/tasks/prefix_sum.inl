#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"

#define PREFIX_SUM_BLOCK_SIZE 1024
#define PREFIX_SUM_WORKGROUP_SIZE PREFIX_SUM_BLOCK_SIZE

struct DispatchIndirectValueCount
{
    DispatchIndirectStruct command;
    daxa_u32 value_count;
};
DAXA_DECL_BUFFER_PTR(DispatchIndirectValueCount)

DAXA_DECL_TASK_HEAD_BEGIN(PrefixSumWriteCommand)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), value_count)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectValueCount), upsweep_command0)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectValueCount), upsweep_command1)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectValueCount), downsweep_command)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(PrefixSumUpsweep)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectValueCount), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), src)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), dst)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), block_sums)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(PrefixSumDownsweep)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectValueCount), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), block_sums)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), values)
DAXA_DECL_TASK_HEAD_END

struct PrefixSumWriteCommandPush
{
    daxa_u32 uint_offset;
};

struct PrefixSumRange
{
    daxa_u32 uint_src_offset;
    daxa_u32 uint_src_stride;
    daxa_u32 uint_dst_offset;
    daxa_u32 uint_dst_stride;
};

struct PrefixSumUpsweepPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    PrefixSumUpsweep uses;
    PrefixSumRange range;
};

struct PrefixSumDownsweepPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    PrefixSumDownsweep uses;
    PrefixSumRange range;
};

#if __cplusplus

#include "../../gpcontext.hpp"
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
    USE_TASK_HEAD(PrefixSumUpsweep)
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
            .compile_options = { .defines = {{"UPSWEEP", "1"}}, },
        },
        .push_constant_size = sizeof(PrefixSumUpsweepPush),
        .name = std::string{PrefixSumUpsweep::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumRange range = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_pipeline(*context->compute_pipelines.at(PrefixSumUpsweep::NAME));
        PrefixSumUpsweepPush push{
            .globals = context->shader_globals_address,
            .range = range,
        };
        ti.copy_task_head_to(&push.uses);
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.command.buffer(),
        });
    }
};

struct PrefixSumDownsweepTask
{
    USE_TASK_HEAD(PrefixSumDownsweep)
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
            .compile_options = {
                .defines = {{"DOWNSWEEP", "1"}},
            },
        },
        .push_constant_size = sizeof(PrefixSumDownsweepPush),
        .name = std::string{PrefixSumDownsweep::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumRange range = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_pipeline(*context->compute_pipelines.at(PrefixSumDownsweep::NAME));
        PrefixSumDownsweepPush push{
            .globals = context->shader_globals_address,
            .range = range,
        };
        ti.copy_task_head_to(&push.uses);
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.command.buffer(),
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
            .value_count = info.value_count_buf,
            .upsweep_command0 = upsweep0_command_buffer,
            .upsweep_command1 = upsweep1_command_buffer,
            .downsweep_command = downsweep_command_buffer,
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
            .command = upsweep0_command_buffer,
            .src = info.src_buf,
            .dst = info.dst_buf,
            .block_sums = block_sums_src,
        },
        .context = info.context,
        .range = {
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
            .command = upsweep1_command_buffer,
            .src = block_sums_src,
            .dst = block_sums_dst,
            .block_sums = total_count,
        },
        .context = info.context,
        .range = {
            .uint_src_offset = 0,
            .uint_src_stride = 1,
            .uint_dst_offset = 0,
            .uint_dst_stride = 1,
        },
    });
    info.task_list.add_task(PrefixSumDownsweepTask{
        .uses={
            .command = downsweep_command_buffer,
            .block_sums = block_sums_dst,
            .values = info.dst_buf,
        },
        .context = info.context,
        .range = {
            .uint_src_offset = std::numeric_limits<u32>::max(),
            .uint_src_stride = std::numeric_limits<u32>::max(),
            .uint_dst_offset = info.dst_uint_offset,
            .uint_dst_stride = info.dst_uint_stride,
        },
    });
}

#endif