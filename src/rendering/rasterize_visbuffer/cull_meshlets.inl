#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/cull_util.inl"

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshlets, 12)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(ShaderGlobals), globals)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletCullArgBucketsBufferHead), meshlets_cull_arg_buckets_buffer)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta_data)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshlet_visibility_bitfield_arena)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(MeshletInstances), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(DrawIndirectStruct), draw_commands)
DAXA_DECL_TASK_HEAD_END

struct CullMeshletsPush
{
    DAXA_TH_BLOB(CullMeshlets, uses)
    daxa_u32 indirect_args_table_id;
    daxa_u32 meshlets_per_indirect_arg;
    daxa_u32 opaque_or_discard;
};

#if __cplusplus
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"

inline static constexpr char const CULL_MESHLETS_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/cull_meshlets.glsl";

inline daxa::ComputePipelineCompileInfo cull_meshlets_pipeline_compile_info()
{
    return {
        .shader_info =
            daxa::ShaderCompileInfo{
                .source = daxa::ShaderFile{CULL_MESHLETS_SHADER_PATH},
                .compile_options = {.defines = {{"CullMeshlets_", "1"}}},
            },
        .push_constant_size = s_cast<u32>(sizeof(CullMeshletsPush) + CullMeshlets::attachment_shader_data_size()),
        .name = std::string{CullMeshlets{}.name()},
    };
};

struct CullMeshletsTask : CullMeshlets
{
    CullMeshlets::AttachmentViews views = {};
    GPUContext * context = {};
    u32 opaque_or_discard = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(CullMeshlets{}.name()));
        for (u32 bucket = 0; bucket < 32; ++bucket)
        {
            ti.recorder.push_constant_vptr({
                .data = ti.attachment_shader_data.data(),
                .size = ti.attachment_shader_data.size(),
            });
            ti.recorder.push_constant(CullMeshletsPush{
                .indirect_args_table_id = bucket,
                .meshlets_per_indirect_arg = (1u << bucket),
                .opaque_or_discard = opaque_or_discard,
            }, CullMeshlets::attachment_shader_data_size());
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.get(CullMeshlets::meshlets_cull_arg_buckets_buffer).ids[0],
                .offset = sizeof(DispatchIndirectStruct) * bucket,
            });
        }
    }
};

#endif