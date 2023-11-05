#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"

#if __cplusplus || defined(FilterVisibleTrianglesWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(FilterVisibleTrianglesWriteCommand, 1)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(MeshletInstances), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus || !defined(FilterVisibleTrianglesWriteCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(FilterVisibleTriangles, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(MeshletInstances), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshlet_visibility_bitfields, daxa_BufferPtr(daxa_u32vec4), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_visible_triangles, daxa_RWBufferPtr(TriangleList), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus

#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const FILTER_VISIBLE_TRIANGLES_PATH[] =
    "./src/rendering/rasterize_visbuffer/filter_visible_triangles.glsl";

using FilterVisibleTrianglesWriteCommandTask = WriteIndirectDispatchArgsBaseTask<
    FilterVisibleTrianglesWriteCommand,
    FILTER_VISIBLE_TRIANGLES_PATH>;

struct FilterVisibleTrianglesTask
{
    DAXA_USE_TASK_HEADER(FilterVisibleTriangles)
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{FILTER_VISIBLE_TRIANGLES_PATH}},
        .name = std::string{FilterVisibleTriangles::NAME},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(FilterVisibleTriangles::NAME));
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