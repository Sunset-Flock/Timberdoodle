#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"

///
/// CullMeshesTask goes through all entities and their meshlists.
/// It checks if the meshes are visible and if they are they get inserted into a visible meshlist.
/// It also generates a list of meshlet counts for each mesh, that the following meshlet culling uses.
///

#define CULL_MESHES_WORKGROUP_X 8
#define CULL_MESHES_WORKGROUP_Y (MAX_MESHES_PER_MESHGROUP)

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshesCommand, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), cull_meshlets_commands)
BUFFER_COMPUTE_WRITE(meshlet_cull_indirect_args, MeshletCullIndirectArgTable)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshes, 11)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroup_indices)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz)
BUFFER_COMPUTE_WRITE(meshlet_cull_indirect_args, MeshletCullIndirectArgTable)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), cull_meshlets_commands)
DAXA_DECL_TASK_HEAD_END

struct CullMeshesCommandPush
{
    DAXA_TH_BLOB(CullMeshesCommand, uses)
    daxa_u32 dummy;
};

struct CullMeshesPush
{
    DAXA_TH_BLOB(CullMeshes, uses)
    daxa_u32 dummy;
};

#if __cplusplus
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshes.glsl";

using CullMeshesCommandWriteTask = WriteIndirectDispatchArgsPushBaseTask<CullMeshesCommand, CULL_MESHES_SHADER_PATH, CullMeshesCommandPush>;
auto cull_meshes_write_command_pipeline_compile_info()
{
    return write_indirect_dispatch_args_base_compile_pipeline_info<CullMeshesCommand, CULL_MESHES_SHADER_PATH, CullMeshesCommandPush>();
}

inline daxa::ComputePipelineCompileInfo cull_meshes_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{CULL_MESHES_SHADER_PATH}},
        .push_constant_size = s_cast<u32>(sizeof(CullMeshesPush) + CullMeshes::attachment_shader_data_size()),
        .name = std::string{CullMeshes{}.name()},
    };
}

struct CullMeshesTask : CullMeshes
{
    CullMeshes::AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(CullMeshes{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(CullMeshes::command).ids[0]});
    }
};

void tasks_cull_meshes(GPUContext * context, daxa::TaskGraph & task_list, CullMeshesTask task)
{
    auto command_buffer = task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "CullMeshesCommand",
    });

    task_list.add_task(CullMeshesCommandWriteTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::globals, daxa::get<daxa::TaskBufferView>(task.views.views.at(CullMeshesTask::globals.value))}},
            daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::entity_meta, daxa::get<daxa::TaskBufferView>(task.views.views.at(CullMeshesTask::entity_meta.value))}},
            daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::command, command_buffer}},
            daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::cull_meshlets_commands, daxa::get<daxa::TaskBufferView>(task.views.views.at(CullMeshesTask::cull_meshlets_commands.value))}},
            daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::meshlet_cull_indirect_args, daxa::get<daxa::TaskBufferView>(task.views.views.at(CullMeshesTask::meshlet_cull_indirect_args.value))}},
        },
        .context = context,
    });

    task.views.views.at(CullMeshesTask::command.value) = command_buffer;
    task_list.add_task(task);
}

#endif