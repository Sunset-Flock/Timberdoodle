#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/scene.inl"
#include "../../../shader_shared/asset.inl"
#include "../../../shader_shared/cull_util.inl"

#if __cplusplus || defined(CullMeshlets_)
DAXA_DECL_TASK_USES_BEGIN(CullMeshlets, 1)
DAXA_TASK_USE_BUFFER(u_commands, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshlet_cull_indirect_args, daxa_BufferPtr(MeshletCullIndirectArgTable), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meta_data, daxa_BufferPtr(GPUEntityMetaData), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshgroups, daxa_BufferPtr(daxa_u32), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshgroups, daxa_BufferPtr(GPUMeshGroup), GRAPHICS_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_combined_transforms, daxa_BufferPtr(daxa_f32mat4x3), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(GPUMesh), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshlet_visibility_bitfield_offsets, EntityMeshletVisibilityBitfieldOffsetsView, COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshlet_visibility_bitfield_arena, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_TASK_USE_IMAGE(u_hiz, REGULAR_2D, COMPUTE_SHADER_SAMPLED)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_RWBufferPtr(MeshletInstances), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_draw_command, daxa_RWBufferPtr(DrawIndirectStruct), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

struct CullMeshletsPush
{
    daxa_u32 indirect_args_table_id;
    daxa_u32 meshlets_per_indirect_arg;
};

#if __cplusplus

#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

inline static constexpr char const CULL_MESHLETS_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshlets.glsl";

struct CullMeshletsTask
{
    DAXA_USE_TASK_HEADER(CullMeshlets)
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{CULL_MESHLETS_SHADER_PATH},
            .compile_options = {.defines = {{"CullMeshlets_", "1"}}},
        },
        .push_constant_size = sizeof(CullMeshletsPush),
        .name = std::string{CullMeshlets::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(CullMeshlets::NAME));
        for (u32 table = 0; table < 32; ++table)
        {
            cmd.push_constant(CullMeshletsPush{
                .indirect_args_table_id = table,
                .meshlets_per_indirect_arg = (1u << table),
            });
            cmd.dispatch_indirect({
                .indirect_buffer = uses.u_commands.buffer(),
                .offset = sizeof(DispatchIndirectStruct) * table,
            });
        }
    }
};

#endif