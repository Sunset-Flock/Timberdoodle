#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"

#define PREFIX_SUM_BLOCK_SIZE 1024
#define PREFIX_SUM_WORKGROUP_SIZE PREFIX_SUM_BLOCK_SIZE

struct DispatchIndirectValueCount
{
    DispatchIndirectStruct command;
    daxa_u32 value_count;
};
DAXA_DECL_BUFFER_PTR(DispatchIndirectValueCount)

DAXA_DECL_TASK_HEAD_BEGIN(PrefixSumCommandWriteH, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), value_count)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectValueCount), upsweep_command0)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectValueCount), upsweep_command1)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectValueCount), downsweep_command)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(PrefixSumUpsweepH, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectValueCount), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), src)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), dst)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), block_sums)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(PrefixSumDownsweepH, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectValueCount), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), block_sums)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), values)
DAXA_DECL_TASK_HEAD_END

struct PrefixSumWriteCommandPush
{
    daxa_u32 uint_offset;
    DAXA_TH_BLOB(PrefixSumCommandWriteH, uses)
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
    PrefixSumRange range;
    DAXA_TH_BLOB(PrefixSumUpsweepH, uses)
};

struct PrefixSumDownsweepPush
{
    PrefixSumRange range;
    DAXA_TH_BLOB(PrefixSumDownsweepH, uses)
};

#if __cplusplus

#include "../../gpu_context.hpp"
#include "misc.hpp"

static constexpr inline char const PREFIX_SUM_SHADER_PATH[] = "./src/rendering/tasks/prefix_sum.glsl";

using PrefixSumCommandWriteTask = 
    SimpleComputeTask< PrefixSumCommandWriteH::Task, PrefixSumWriteCommandPush, PREFIX_SUM_SHADER_PATH, "main">;

// Sums n <= 1024 values up. Always writes 1024 values out (for simplicity in multi pass).
inline daxa::ComputePipelineCompileInfo prefix_sum_upsweep_pipeline_compile_info()
{
    return {
        .shader_info =
            daxa::ShaderCompileInfo{
                .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
                .compile_options = {.defines = {{"UPSWEEP", "1"}}},
            },
        .push_constant_size = s_cast<u32>(sizeof(PrefixSumUpsweepPush)),
        .name = std::string{PrefixSumUpsweepH::NAME},
    };
}
struct PrefixSumUpsweepTask : PrefixSumUpsweepH::Task
{
    AttachmentViews views = {};
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumRange range = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(prefix_sum_upsweep_pipeline_compile_info().name));
        PrefixSumUpsweepPush push{ .range = range };
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(AT.command).ids[0]});
    }
};

inline daxa::ComputePipelineCompileInfo prefix_sum_downsweep_pipeline_compile_info()
{
    return {
        .shader_info =
            daxa::ShaderCompileInfo{
                .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
                .compile_options = {.defines = {{"DOWNSWEEP", "1"}}},
            },
        .push_constant_size = s_cast<u32>(sizeof(PrefixSumDownsweepPush)),
        .name = std::string{PrefixSumDownsweepH::NAME},
    };
}
struct PrefixSumDownsweepTask : PrefixSumDownsweepH::Task
{
    AttachmentViews views = {};
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumRange range = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(prefix_sum_upsweep_pipeline_compile_info().name));
        PrefixSumUpsweepPush push{ .range = range };
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(AT.command).ids[0]});
    }
};

// Task function that can prefix sum up to 2^20 million values.
// Reads values from src buffer with src_offset and src_stride,
// writes values to dst buffer with dst_offset and dst_stride.
struct PrefixSumTaskGroupInfo
{
    RenderContext * render_context;
    daxa::TaskGraph & task_list;
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
        .views = std::array{
            daxa::attachment_view(PrefixSumCommandWriteH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(PrefixSumCommandWriteH::AT.value_count, info.value_count_buf),
            daxa::attachment_view(PrefixSumCommandWriteH::AT.upsweep_command0, upsweep0_command_buffer),
            daxa::attachment_view(PrefixSumCommandWriteH::AT.upsweep_command1, upsweep1_command_buffer),
            daxa::attachment_view(PrefixSumCommandWriteH::AT.downsweep_command, downsweep_command_buffer),
        },
        .context = info.render_context->gpuctx,
        .push = {.uint_offset = info.value_count_uint_offset},
    });

    auto max_block_count = (static_cast<u64>(info.max_value_count) + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
    auto block_sums_src = info.task_list.create_transient_buffer({
        .size = static_cast<u32>(sizeof(u32) * max_block_count),
        .name = "prefix sum block_sums_src",
    });

    info.task_list.add_task(PrefixSumUpsweepTask{
        .views = std::array{
            daxa::attachment_view(PrefixSumUpsweepH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(PrefixSumUpsweepH::AT.command, upsweep0_command_buffer),
            daxa::attachment_view(PrefixSumUpsweepH::AT.src, info.src_buf),
            daxa::attachment_view(PrefixSumUpsweepH::AT.dst, info.dst_buf),
            daxa::attachment_view(PrefixSumUpsweepH::AT.block_sums, block_sums_src),
        },
        .context = info.render_context->gpuctx,
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
        .views = std::array{
            daxa::attachment_view(PrefixSumUpsweepH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(PrefixSumUpsweepH::AT.command, upsweep1_command_buffer),
            daxa::attachment_view(PrefixSumUpsweepH::AT.src, block_sums_src),
            daxa::attachment_view(PrefixSumUpsweepH::AT.dst, block_sums_dst),
            daxa::attachment_view(PrefixSumUpsweepH::AT.block_sums, total_count),
        },
        .context = info.render_context->gpuctx,
        .range = {
            .uint_src_offset = 0,
            .uint_src_stride = 1,
            .uint_dst_offset = 0,
            .uint_dst_stride = 1,
        },
    });

    info.task_list.add_task(PrefixSumDownsweepTask{
        .views = std::array{
            daxa::attachment_view(PrefixSumDownsweepH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(PrefixSumDownsweepH::AT.command, downsweep_command_buffer),
            daxa::attachment_view(PrefixSumDownsweepH::AT.block_sums, block_sums_dst),
            daxa::attachment_view(PrefixSumDownsweepH::AT.values, info.dst_buf),
        },
        .context = info.render_context->gpuctx,
        .range = {
            .uint_src_offset = std::numeric_limits<u32>::max(),
            .uint_src_stride = std::numeric_limits<u32>::max(),
            .uint_dst_offset = info.dst_uint_offset,
            .uint_dst_stride = info.dst_uint_stride,
        },
    });
}

#endif