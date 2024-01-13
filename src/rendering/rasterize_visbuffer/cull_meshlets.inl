#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/asset.inl"
#include "../../shader_shared/cull_util.inl"

DAXA_DECL_TASK_HEAD_BEGIN(CullMeshlets, 12)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, hiz)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), commands)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletCullIndirectArgTable), meshlet_cull_indirect_args)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUEntityMetaData), entity_meta_data)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), meshgroups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), entity_combined_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(
    COMPUTE_SHADER_READ, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_meshlet_visibility_bitfield_arena)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(DrawIndirectStruct), draw_command)
DAXA_DECL_TASK_HEAD_END

struct CullMeshletsPush
{
    DAXA_TH_BLOB(CullMeshlets) uses;
    daxa_BufferPtr(ShaderGlobals) globals;
    daxa_u32 indirect_args_table_id;
    daxa_u32 meshlets_per_indirect_arg;
};

#if __cplusplus
#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"

inline static constexpr char const CULL_MESHLETS_SHADER_PATH[] =
    "./src/rendering/rasterize_visbuffer/cull_meshlets.glsl";

struct CullMeshletsTask : CullMeshlets
{
    GPUContext *context = {};

    inline static daxa::ComputePipelineCompileInfo const PIPELINE_COMPILE_INFO{
        .shader_info =
            daxa::ShaderCompileInfo{
                .source = daxa::ShaderFile{CULL_MESHLETS_SHADER_PATH},
                .compile_options = {.defines = {{"CullMeshlets_", "1"}}},
            },
        .push_constant_size = sizeof(CullMeshletsPush),
        .name = std::string{CullMeshlets{}.name()},
    };

    virtual void callback(daxa::TaskInterface ti) const override
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(CullMeshlets{}.name()));
        for (u32 table = 0; table < 32; ++table)
        {
            auto push = CullMeshletsPush{
                .uses = span_to_array<DAXA_TH_BLOB(CullMeshlets){}.size()>(ti.attachment_shader_data_blob),
                .globals = context->shader_globals_address,
                .indirect_args_table_id = table,
                .meshlets_per_indirect_arg = (1u << table),
            };
            ti.recorder.push_constant(push);
            ti.recorder.dispatch_indirect({
                .indirect_buffer = ti.buf(commands).ids[0],
                .offset = sizeof(DispatchIndirectStruct) * table,
            });
        }
    }
};

#endif