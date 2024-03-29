#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/cull_util.inl"
#include "../../shader_shared/geometry_pipeline.inl"

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshletsH, 12)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletCullArgBucketsBufferHead), meshlets_cull_arg_buckets_buffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta_data)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), first_pass_meshlets_bitfield_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, U32ArenaBufferRef, first_pass_meshlets_bitfield_arena)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), draw_commands)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshlets2H, 12)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletCullArgBucketsBufferHead), meshlets_cull_arg_buckets_buffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta_data)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), first_pass_meshlets_bitfield_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, U32ArenaBufferRef, first_pass_meshlets_bitfield_arena)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(daxa_u32), draw_commands)
DAXA_DECL_TASK_HEAD_END

struct CullMeshletsPush
{
    DAXA_TH_BLOB(CullMeshletsH, uses)
    daxa_u32 indirect_args_table_id;
    daxa_u32 meshlets_per_indirect_arg;
    daxa_u32 draw_list_type;
};

#if defined(__cplusplus)
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"

using CullMeshletsTask2 = SimpleComputeTask<
    CullMeshlets2H::Task, 
    CullMeshletsPush, 
    "./src/rendering/rasterize_visbuffer/cull_meshlets.slang",
    "entry_cull_meshlets"
>;

inline static constexpr char const CULL_MESHLETS_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/cull_meshlets.glsl";
inline static constexpr char const SLANG_CULL_MESHLETS_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/cull_meshlets.slang";

SANE_STATIC_BEGIN(cull_meshlets_pipeline_compile_info)
daxa::ComputePipelineCompileInfo{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{CULL_MESHLETS_SHADER_PATH},
        .compile_options = {.defines = {{"CullMeshlets_", "1"}}},
    },
    .push_constant_size = s_cast<u32>(sizeof(CullMeshletsPush)),
    .name = std::string{CullMeshletsH::NAME},
};
SANE_STATIC_END

SANE_STATIC_BEGIN(slang_cull_meshlets_pipeline_compile_info)
daxa::ComputePipelineCompileInfo{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{SLANG_CULL_MESHLETS_SHADER_PATH},
        .compile_options = {
            .entry_point = "entry_cull_meshlets",
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {{"CullMeshlets_", "1"}},
        },
    },
    .push_constant_size = s_cast<u32>(sizeof(CullMeshletsPush)),
    .name = std::string{CullMeshlets2H::NAME},
};
SANE_STATIC_END

struct CullMeshletsTask : CullMeshletsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        if (render_context->render_data.settings.use_slang_for_culling)
        {
            ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(slang_cull_meshlets_pipeline_compile_info().name));
        }
        else
        {
            ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(cull_meshlets_pipeline_compile_info().name));
        }
        for (u32 draw_list_type = 0; draw_list_type < DRAW_LIST_TYPES; ++draw_list_type)
        {
            for (u32 bucket = 0; bucket < 32; ++bucket)
            {
                CullMeshletsPush push = {
                    .indirect_args_table_id = bucket,
                    .meshlets_per_indirect_arg = (1u << bucket),
                    .draw_list_type = draw_list_type,
                };
                assign_blob(push.uses, ti.attachment_shader_blob);
                ti.recorder.push_constant(push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(AT.meshlets_cull_arg_buckets_buffer).ids[0],
                    .offset = sizeof(DispatchIndirectStruct) * bucket + sizeof(CullMeshletsArgBuckets) * draw_list_type,
                });
            }
        }
    }
};

#endif