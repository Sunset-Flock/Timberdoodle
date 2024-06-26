#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"

#if __cplusplus || defined(FilterVisibleTrianglesWriteCommand_COMMAND)
DAXA_DECL_TASK_HEAD_BEGIN(FilterVisibleTrianglesWriteCommand)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), u_instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), u_command)
DAXA_DECL_TASK_HEAD_END
#endif

#if __cplusplus || !defined(FilterVisibleTrianglesWriteCommand_COMMAND)
DAXA_DECL_TASK_HEAD_BEGIN(FilterVisibleTriangles)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), u_command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), u_instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32vec4), u_meshlet_visibility_bitfields)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(TriangleList), u_visible_triangles)
DAXA_DECL_TASK_HEAD_END
#endif

#if __cplusplus

#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const FILTER_VISIBLE_TRIANGLES_PATH[] =
    "./src/rendering/rasterize_visbuffer/filter_visible_triangles.glsl";

using FilterVisibleTrianglesWriteCommandTask = WriteIndirectDispatchArgsBaseTask<
    FilterVisibleTrianglesWriteCommand,
    FILTER_VISIBLE_TRIANGLES_PATH>;

struct FilterVisibleTrianglesTask : FilterVisibleTriangles
{
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{FILTER_VISIBLE_TRIANGLES_PATH}},
        .name = std::string{FilterVisibleTriangles{}.name()},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(FilterVisibleTriangles{}.name()));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

void task_filter_visible_triangles(GPUContext * context, daxa::TaskGraph & task_graph, FilterVisibleTriangles::Uses uses)
{
    auto command_buffer = task_graph.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "task_filter_visible_triangles command_buffer",
    });
    task_graph.add_task(FilterVisibleTrianglesWriteCommandTask{
        .uses={
            .u_instantiated_meshlets = uses.u_instantiated_meshlets,
            .u_command = command_buffer,
        },
        .context = context,
    });
    uses.u_command.handle = command_buffer;
    task_graph.add_task(FilterVisibleTrianglesTask{
        .uses=uses,
        .context=context,
    });
}

#endif