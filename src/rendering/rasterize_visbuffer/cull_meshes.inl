#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"

///
/// CullMeshesTask goes through all entities and their meshlists.
/// It checks if the meshes are visible and if they are they get inserted into a visible meshlist.
/// It also generates a list of meshlet counts for each mesh, that the following meshlet culling uses.
///

#define CULL_MESHES_WORKGROUP_X 8
#define CULL_MESHES_WORKGROUP_Y 7

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshesCommand, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), cull_meshlets_commands)
BUFFER_COMPUTE_WRITE(meshlet_cull_indirect_args, MeshletCullIndirectArgTable)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshes, 10)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroup_indices)
DAXA_TH_BUFFER_PTR(GRAPHICS_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz)
BUFFER_COMPUTE_WRITE(meshlet_cull_indirect_args, MeshletCullIndirectArgTable)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), cull_meshlets_commands)
DAXA_DECL_TASK_HEAD_END

struct CullMeshesCommandPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DAXA_TH_BLOB(CullMeshesCommand) uses;
};

struct CullMesheshPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DAXA_TH_BLOB(CullMeshes) uses;
};

#if __cplusplus
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshes.glsl";

using CullMeshesCommandWriteTask =
    WriteIndirectDispatchArgsPushBaseTask<CullMeshesCommand, CULL_MESHES_SHADER_PATH, CullMeshesCommandPush>;

struct CullMeshesTask : CullMeshes
{
    static inline daxa::ComputePipelineCompileInfo const PIPELINE_COMPILE_INFO = {
        .shader_info =
            daxa::ShaderCompileInfo{
                .source = daxa::ShaderFile{CULL_MESHES_SHADER_PATH},
            },
        .push_constant_size = sizeof(CullMesheshPush),
        .name = std::string{CullMeshes{}.name()},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(CullMeshes{}.name()));
        CullMesheshPush push = {
            .globals = context->shader_globals_address,
            .uses = span_to_array<DAXA_TH_BLOB(CullMeshes){}.size()>(ti.shader_byte_blob),
        };
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.buf_attach(command).ids[0]});
    }
};

// TODO(msakmary) PATRICK REVIEW
void tasks_cull_meshes(GPUContext * context, daxa::TaskGraph & task_list, decltype(CullMeshes::attachments) uses)
{
    auto command_buffer = task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "CullMeshesCommand",
    });
    CullMeshesTask cull_task = {};

    CullMeshesCommandWriteTask write_task = {};
    write_task.attachments.set_view(write_task.entity_meta, uses[cull_task.entity_meta].view);
    write_task.attachments.set_view(write_task.command, uses[cull_task.command].view);
    write_task.attachments.set_view(write_task.cull_meshlets_commands, uses[cull_task.cull_meshlets_commands].view);
    write_task.attachments.set_view(write_task.meshlet_cull_indirect_args, uses[cull_task.meshlet_cull_indirect_args].view);
    write_task.context = context;
    task_list.add_task(write_task);

    cull_task.attachments = uses;
    cull_task.attachments.set_view(cull_task.command, command_buffer);
    cull_task.context = context;
    task_list.add_task(cull_task);
}

#endif