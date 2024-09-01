#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/po2_expansion.inl"

#define CULL_MESHES_WORKGROUP_X 128

DAXA_DECL_TASK_HEAD_BEGIN(ExpandMeshesToMeshletsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMaterial), materials)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroup_indices)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz) // OPTIONAL
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hip) // OPTIONAL
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(Po2WorkExpansionBufferHead), opaque_po2expansion)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(Po2WorkExpansionBufferHead), masked_opaque_po2expansion)
// TODO REMOVE, PUT IN VSM GLOBALS
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_DECL_TASK_HEAD_END

struct ExpandMeshesToMeshletsPush
{
    DAXA_TH_BLOB(ExpandMeshesToMeshletsH, uses)
    daxa::b32 cull_meshes;
    // Only used for vsms:
    daxa::u32 cascade;
};

#if defined(__cplusplus)
#include "../../gpu_context.hpp"
#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshes.hlsl";

inline daxa::ComputePipelineCompileInfo expand_meshes_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_MESHES_SHADER_PATH},
            .compile_options = {
                .entry_point = "main",
                .language = daxa::ShaderLanguage::SLANG,
            },
        },
        .push_constant_size = s_cast<u32>(sizeof(ExpandMeshesToMeshletsPush)),
        .name = std::string{ExpandMeshesToMeshletsH::NAME},
    };
}

struct ExpandMeshesToMeshletsTask : ExpandMeshesToMeshletsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    bool cull_meshes = {};
    // only used for vsm cull:
    u32 cascade = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(expand_meshes_pipeline_compile_info().name));
        ExpandMeshesToMeshletsPush push = {
            .cull_meshes = cull_meshes,
            .cascade = cascade,
        };
        assign_blob(push.uses, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        auto total_mesh_draws =
            render_context->mesh_instance_counts.prepass_instance_counts[0] +
            render_context->mesh_instance_counts.prepass_instance_counts[1];
        total_mesh_draws = std::min(total_mesh_draws, MAX_MESH_INSTANCES);
        ti.recorder.dispatch(daxa::DispatchInfo{round_up_div(total_mesh_draws, CULL_MESHES_WORKGROUP_X), 1, 1});
    }
};

struct TaskExpandMeshesToMeshletsInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & task_list;
    bool cull_meshes = {};
    // Used for VSM page culling:
    daxa::TaskImageView vsm_hip = daxa::NullTaskImage;
    daxa::u32 vsm_cascade = {};
    daxa::TaskBufferView vsm_clip_projections = daxa::NullTaskBuffer;
    daxa::TaskImageView hiz = daxa::NullTaskImage;
    daxa::TaskBufferView globals = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView materials = {};
    daxa::TaskBufferView entity_meta = {};
    daxa::TaskBufferView entity_meshgroup_indices = {};
    daxa::TaskBufferView meshgroups = {};
    daxa::TaskBufferView entity_transforms = {};
    daxa::TaskBufferView entity_combined_transforms = {};
    std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPES> & opaque_meshlet_cull_po2expansions;
    DispatchIndirectStruct dispatch_clear = {0, 1, 1};
    std::string buffer_name_prefix = "";
};
void tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo const & info)
{
    auto opaque_po2expansion = info.task_list.create_transient_buffer({
        .size = Po2WorkExpansionBufferHead::calc_buffer_size(MAX_MESH_INSTANCES, MAX_MESHLET_INSTANCES),
        .name = info.buffer_name_prefix + "opaque_meshlet_cull_po2expansion",
    });
    auto masked_opaque_po2expansion = info.task_list.create_transient_buffer({
        .size = Po2WorkExpansionBufferHead::calc_buffer_size(MAX_MESH_INSTANCES, MAX_MESHLET_INSTANCES),
        .name = info.buffer_name_prefix + "masked_opaque_meshlet_cull_po2expansion",
    });
    info.task_list.add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, opaque_po2expansion),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, masked_opaque_po2expansion),
        },
        .task = [=](daxa::TaskInterface ti)
        {
            allocate_fill_copy(
                ti, Po2WorkExpansionBufferHead::create(ti.device.buffer_device_address(ti.get(opaque_po2expansion).ids[0]).value(), MAX_MESH_INSTANCES, MAX_MESHLET_INSTANCES, 32, info.dispatch_clear),
                ti.get(opaque_po2expansion));
            allocate_fill_copy(
                ti, Po2WorkExpansionBufferHead::create(ti.device.buffer_device_address(ti.get(masked_opaque_po2expansion).ids[0]).value(), MAX_MESH_INSTANCES, MAX_MESHLET_INSTANCES, 32, info.dispatch_clear),
                ti.get(masked_opaque_po2expansion));
        },
        .name = "init meshlet cull arg buckets buffer",
    });

    info.task_list.add_task(ExpandMeshesToMeshletsTask{
        .views = std::array{
            ExpandMeshesToMeshletsH::AT.globals | info.globals,
            ExpandMeshesToMeshletsH::AT.mesh_instances | info.mesh_instances,
            ExpandMeshesToMeshletsH::AT.meshes | info.meshes,
            ExpandMeshesToMeshletsH::AT.materials | info.materials,
            ExpandMeshesToMeshletsH::AT.entity_meta | info.entity_meta,
            ExpandMeshesToMeshletsH::AT.entity_meshgroup_indices | info.entity_meshgroup_indices,
            ExpandMeshesToMeshletsH::AT.meshgroups | info.meshgroups,
            ExpandMeshesToMeshletsH::AT.entity_transforms | info.entity_transforms,
            ExpandMeshesToMeshletsH::AT.entity_combined_transforms | info.entity_combined_transforms,
            ExpandMeshesToMeshletsH::AT.opaque_po2expansion | opaque_po2expansion,
            ExpandMeshesToMeshletsH::AT.masked_opaque_po2expansion | masked_opaque_po2expansion,
            ExpandMeshesToMeshletsH::AT.hiz | info.hiz,
            ExpandMeshesToMeshletsH::AT.hip | info.vsm_hip,
            ExpandMeshesToMeshletsH::AT.vsm_clip_projections | info.vsm_clip_projections,
        },
        .render_context = info.render_context,
        .cull_meshes = info.cull_meshes,
        .cascade = info.vsm_cascade,
    });
    info.opaque_meshlet_cull_po2expansions = std::array{opaque_po2expansion, masked_opaque_po2expansion};
}

#endif