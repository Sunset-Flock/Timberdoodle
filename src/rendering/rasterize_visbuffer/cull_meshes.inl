#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/geometry_pipeline.inl"

///
/// CullMeshesTask goes through all entities and their meshlists.
/// It checks if the meshes are visible and if they are they get inserted into a visible meshlist.
/// It also generates a list of meshlet counts for each mesh, that the following meshlet culling uses.
///

#define CULL_MESHES_WORKGROUP_X 128
#define CULL_MESHES_WORKGROUP_Y 1

// DAXA_DECL_TASK_HEAD_BEGIN(CullMeshesCommand, 3)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), command)
// DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshes, 12)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(ShaderGlobals), globals)
// DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_u64, command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(OpaqueMeshDrawListBufferHead), opaque_mesh_draw_lists)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMaterial), materials)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroup_indices)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(
    COMPUTE_SHADER_READ_WRITE, 
    daxa_RWBufferPtr(MeshletCullArgBucketsBufferHead), 
    meshlet_cull_arg_buckets_opaque)
DAXA_TH_BUFFER_PTR(
    COMPUTE_SHADER_READ_WRITE, 
    daxa_RWBufferPtr(MeshletCullArgBucketsBufferHead), 
    meshlet_cull_arg_buckets_discard)
DAXA_DECL_TASK_HEAD_END

// struct CullMeshesCommandPush
// {
//     DAXA_TH_BLOB(CullMeshesCommand, uses)
//     daxa_u32 dummy;
// };

struct CullMeshesPush
{
    DAXA_TH_BLOB(CullMeshes, uses)
    daxa_u32 dummy;
};

#if __cplusplus
#include "../../gpu_context.hpp"
#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshes.glsl";

// using CullMeshesCommandWriteTask = WriteIndirectDispatchArgsPushBaseTask<CullMeshesCommand, CULL_MESHES_SHADER_PATH, CullMeshesCommandPush>;
// auto cull_meshes_write_command_pipeline_compile_info()
// {
//     return write_indirect_dispatch_args_base_compile_pipeline_info<CullMeshesCommand, CULL_MESHES_SHADER_PATH, CullMeshesCommandPush>();
// }

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
    AttachmentViews views = {};
    GPUContext * context = {};
    SceneRendererContext* scene_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(CullMeshes{}.name()));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        auto const total_mesh_draws = scene_context->opaque_draw_lists[0].size() + scene_context->opaque_draw_lists[1].size();
        ti.recorder.dispatch(daxa::DispatchInfo{
            round_up_div(total_mesh_draws, CULL_MESHES_WORKGROUP_X),
            1,
            1,
        });
    }
};

struct TaskCullMeshesInfo
{
    GPUContext * context = {};
    SceneRendererContext * scene_context = {};
    daxa::TaskGraph & task_list;
    daxa::TaskBufferView globals = {};
    daxa::TaskBufferView opaque_draw_lists = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView materials = {};
    daxa::TaskBufferView entity_meta = {};
    daxa::TaskBufferView entity_meshgroup_indices = {};
    daxa::TaskBufferView meshgroups = {};
    daxa::TaskBufferView entity_transforms = {};
    daxa::TaskBufferView entity_combined_transforms = {};
    daxa::TaskImageView hiz = {};
};
auto tasks_cull_meshes(TaskCullMeshesInfo const & info) -> std::pair<daxa::TaskBufferView, daxa::TaskBufferView>
{
    auto meshlets_cull_arg_buckets_buffer_opaque = info.task_list.create_transient_buffer({
        .size = meshlet_cull_arg_buckets_buffer_size(MAX_MESH_INSTANCES, MAX_MESHLET_INSTANCES),
        .name = "meshlet_cull_arg_buckets_buffer_opaque",
    });
    auto meshlets_cull_arg_buckets_buffer_discard = info.task_list.create_transient_buffer({
        .size = meshlet_cull_arg_buckets_buffer_size(MAX_MESH_INSTANCES, MAX_MESHLET_INSTANCES),
        .name = "meshlet_cull_arg_buckets_buffer_discard",
    });

    info.task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, meshlets_cull_arg_buckets_buffer_opaque),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, meshlets_cull_arg_buckets_buffer_discard),
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto alloc0 = ti.allocator->allocate_fill(
                meshlet_cull_arg_buckets_buffer_make_head(
                    MAX_MESH_INSTANCES, 
                    MAX_MESHLET_INSTANCES, 
                    ti.device.get_device_address(ti.get(meshlets_cull_arg_buckets_buffer_opaque).ids[0]).value())).value();
            auto alloc1 = ti.allocator->allocate_fill(
                meshlet_cull_arg_buckets_buffer_make_head(
                    MAX_MESH_INSTANCES, 
                    MAX_MESHLET_INSTANCES, 
                    ti.device.get_device_address(ti.get(meshlets_cull_arg_buckets_buffer_discard).ids[0]).value())).value();
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.allocator->buffer(),
                .dst_buffer = ti.get(meshlets_cull_arg_buckets_buffer_opaque).ids[0],
                .src_offset = alloc0.buffer_offset,
                .size = sizeof(MeshletCullArgBucketsBufferHead),
            });            
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = ti.allocator->buffer(),
                .dst_buffer = ti.get(meshlets_cull_arg_buckets_buffer_discard).ids[0],
                .src_offset = alloc1.buffer_offset,
                .size = sizeof(MeshletCullArgBucketsBufferHead),
            });
        },
        .name = "init meshlet cull arg buckets buffer",
    });

    // auto command_buffer = info.task_list.create_transient_buffer({
    //     .size = sizeof(DispatchIndirectStruct),
    //     .name = "CullMeshesCommand",
    // });

    // info.task_list.add_task(CullMeshesCommandWriteTask{
    //     .views = std::array{
    //         daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::globals, info.globals}},
    //         daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::entity_meta, info.entity_meta}},
    //         daxa::TaskViewVariant{std::pair{CullMeshesCommandWriteTask::command, command_buffer}},
    //     },
    //     .context = info.context,
    // });

    info.task_list.add_task(CullMeshesTask{
        .views = std::array{
            daxa::attachment_view(CullMeshesTask::globals, info.globals),
            daxa::attachment_view(CullMeshesTask::opaque_mesh_draw_lists, info.opaque_draw_lists),
            // daxa::attachment_view(CullMeshesTask::command, command_buffer),
            daxa::attachment_view(CullMeshesTask::meshes, info.meshes),
            daxa::attachment_view(CullMeshesTask::materials, info.materials),
            daxa::attachment_view(CullMeshesTask::entity_meta, info.entity_meta),
            daxa::attachment_view(CullMeshesTask::entity_meshgroup_indices, info.entity_meshgroup_indices),
            daxa::attachment_view(CullMeshesTask::meshgroups, info.meshgroups),
            daxa::attachment_view(CullMeshesTask::entity_transforms, info.entity_transforms),
            daxa::attachment_view(CullMeshesTask::entity_combined_transforms, info.entity_combined_transforms),
            daxa::attachment_view(CullMeshesTask::hiz, info.hiz),
            daxa::attachment_view(CullMeshesTask::meshlet_cull_arg_buckets_opaque, meshlets_cull_arg_buckets_buffer_opaque),
            daxa::attachment_view(CullMeshesTask::meshlet_cull_arg_buckets_discard, meshlets_cull_arg_buckets_buffer_discard),
        },
        .context = info.context,
        .scene_context = info.scene_context,
    });
    return {meshlets_cull_arg_buckets_buffer_opaque, meshlets_cull_arg_buckets_buffer_discard};
}

#endif