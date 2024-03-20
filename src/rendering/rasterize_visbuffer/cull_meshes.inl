#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/geometry_pipeline.inl"

#define CULL_MESHES_WORKGROUP_X 128

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshes, 12)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
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

struct CullMeshesPush
{
    DAXA_TH_BLOB(CullMeshes, uses)
    daxa_u32 dummy;
};

#if defined(__cplusplus)
#include "../../gpu_context.hpp"
#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshes.glsl";

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
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(cull_meshes_pipeline_compile_info().name));
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
        });
        auto const total_mesh_draws = render_context->scene_draw.opaque_draw_lists[0].size() + render_context->scene_draw.opaque_draw_lists[1].size();
        ti.recorder.dispatch(daxa::DispatchInfo{
            round_up_div(total_mesh_draws, CULL_MESHES_WORKGROUP_X),
            1,
            1,
        });
    }
};

struct TaskCullMeshesInfo
{
    RenderContext * render_context = {};
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
    daxa::TaskBufferView & meshlets_cull_arg_buckets_buffer_opaque;
    daxa::TaskBufferView & meshlets_cull_arg_buckets_buffer_discard;
};
void tasks_cull_meshes(TaskCullMeshesInfo const & info)
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
            auto const opaque_solid_head = meshlet_cull_arg_buckets_buffer_make_head(
                    MAX_MESH_INSTANCES, 
                    MAX_MESHLET_INSTANCES, 
                    ti.device.get_device_address(ti.get(meshlets_cull_arg_buckets_buffer_opaque).ids[0]).value());
            allocate_fill_copy(ti, opaque_solid_head, ti.get(meshlets_cull_arg_buckets_buffer_opaque));
            auto const opaque_masked_head = meshlet_cull_arg_buckets_buffer_make_head(
                    MAX_MESH_INSTANCES, 
                    MAX_MESHLET_INSTANCES, 
                    ti.device.get_device_address(ti.get(meshlets_cull_arg_buckets_buffer_discard).ids[0]).value());
            allocate_fill_copy(ti, opaque_masked_head, ti.get(meshlets_cull_arg_buckets_buffer_discard));
        },
        .name = "init meshlet cull arg buckets buffer",
    });

    info.task_list.add_task(CullMeshesTask{
        .views = std::array{
            daxa::attachment_view(CullMeshesTask::globals, info.globals),
            daxa::attachment_view(CullMeshesTask::opaque_mesh_draw_lists, info.opaque_draw_lists),
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
        .render_context = info.render_context,
    });
    info.meshlets_cull_arg_buckets_buffer_opaque = meshlets_cull_arg_buckets_buffer_opaque;
    info.meshlets_cull_arg_buckets_buffer_discard = meshlets_cull_arg_buckets_buffer_discard;
}

#endif