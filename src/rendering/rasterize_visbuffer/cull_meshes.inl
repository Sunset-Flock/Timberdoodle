#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/asset.inl"
#include "../../../shader_shared/scene.inl"

/// 
/// CullMeshesTask goes through all entities and their meshlists.
/// It checks if the meshes are visible and if they are they get inserted into a visible meshlist.
/// It also generates a list of meshlet counts for each mesh, that the following meshlet culling uses.
///

#define CULL_MESHES_WORKGROUP_X 8
#define CULL_MESHES_WORKGROUP_Y 7

#if __cplusplus || defined(CullMeshesCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(CullMeshesCommand, 1)
DAXA_TASK_USE_BUFFER(u_entity_meta, daxa_BufferPtr(GPUEntityMetaData), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_TASK_USE_BUFFER(u_cull_meshlets_commands, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_WRITE)
BUFFER_COMPUTE_WRITE(u_meshlet_cull_indirect_args, MeshletCullIndirectArgTable)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || !defined(CullMeshesCommand_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(CullMeshes, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(GPUMesh), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meta, daxa_BufferPtr(GPUEntityMetaData), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshgroup_indices, daxa_BufferPtr(daxa_u32), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshgroups, daxa_BufferPtr(GPUMeshGroup), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_transforms, daxa_BufferPtr(daxa_f32mat4x3), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_combined_transforms, daxa_BufferPtr(daxa_f32mat4x3), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(u_hiz, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
BUFFER_COMPUTE_WRITE(u_meshlet_cull_indirect_args, MeshletCullIndirectArgTable)
DAXA_TASK_USE_BUFFER(u_cull_meshlets_commands, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus
#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/cull_meshes.glsl";

using CullMeshesCommandWriteTask = WriteIndirectDispatchArgsBaseTask<
    CullMeshesCommand,
    CULL_MESHES_SHADER_PATH>;

struct CullMeshesTask
{
    DAXA_USE_TASK_HEADER(CullMeshes)
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_MESHES_SHADER_PATH},
        },
        .name = std::string{CullMeshes::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(CullMeshes::NAME));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

void tasks_cull_meshes(GPUContext * context, daxa::TaskGraph& task_list, CullMeshes::Uses uses)
{
    auto command_buffer = task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "CullMeshesCommand",
    });

    task_list.add_task(CullMeshesCommandWriteTask{
        .uses={
            .u_entity_meta = uses.u_entity_meta,
            .u_command = command_buffer,
            .u_cull_meshlets_commands = uses.u_cull_meshlets_commands.handle,
            .u_meshlet_cull_indirect_args = uses.u_meshlet_cull_indirect_args,
        },
        .context = context,
    });

    uses.u_command.handle = command_buffer;

    task_list.add_task(CullMeshesTask{
        .uses={uses},
        .context = context,
    });
}

#endif