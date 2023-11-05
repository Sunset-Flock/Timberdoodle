#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/asset.inl"
#define PREPOPULATE_INST_MESHLETS_X 256

#if __cplusplus || defined(PrepopulateInstantiatedMeshletsCommandWrite_COMMAND)
DAXA_DECL_TASK_USES_BEGIN(PrepopulateInstantiatedMeshletsCommandWrite, 1)
DAXA_TASK_USE_BUFFER(u_visible_meshlets_prev, daxa_BufferPtr(VisibleMeshletList), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_command, daxa_RWBufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_WRITE)
DAXA_DECL_TASK_USES_END()
#endif
#if __cplusplus || !defined(PrepopulateInstantiatedMeshletsCommandWrite_COMMAND) && !defined(SetEntityMeshletVisibilityBitMasks_SHADER)
// In the future we should check if the entity slot is actually valid here.
// To do that we need a version in the entity id and a version table we can compare to
DAXA_DECL_TASK_USES_BEGIN(PrepopulateInstantiatedMeshlets, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_visible_meshlets_prev, daxa_BufferPtr(VisibleMeshletList), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets_prev, daxa_BufferPtr(MeshletInstances), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_meshes, daxa_BufferPtr(GPUMesh), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_RWBufferPtr(MeshletInstances), COMPUTE_SHADER_READ_WRITE)
DAXA_TASK_USE_BUFFER(u_entity_meshlet_visibility_bitfield_offsets, EntityMeshletVisibilityBitfieldOffsetsView, COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus || defined(SetEntityMeshletVisibilityBitMasks_SHADER)
DAXA_DECL_TASK_USES_BEGIN(SetEntityMeshletVisibilityBitMasks, 1)
DAXA_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectStruct), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_instantiated_meshlets, daxa_BufferPtr(MeshletInstances), COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshlet_visibility_bitfield_offsets, EntityMeshletVisibilityBitfieldOffsetsView, COMPUTE_SHADER_READ)
DAXA_TASK_USE_BUFFER(u_entity_meshlet_visibility_bitfield_arena, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ_WRITE)
DAXA_DECL_TASK_USES_END()
#endif

#if __cplusplus

#include "../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const PRE_POPULATE_INST_MESHLETS_PATH[] =
    "./src/rendering/rasterize_visbuffer/prepopulate_inst_meshlets.glsl";

using PrepopulateInstantiatedMeshletsCommandWriteTask = WriteIndirectDispatchArgsBaseTask<
    PrepopulateInstantiatedMeshletsCommandWrite,
    PRE_POPULATE_INST_MESHLETS_PATH>;

struct PrepopulateInstantiatedMeshletsTask
{
    DAXA_USE_TASK_HEADER(PrepopulateInstantiatedMeshlets)
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_INST_MESHLETS_PATH}},
        .name = std::string{PrepopulateInstantiatedMeshlets::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(PrepopulateInstantiatedMeshlets::NAME));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

struct SetEntityMeshletVisibilityBitMasksTask
{
    DAXA_USE_TASK_HEADER(SetEntityMeshletVisibilityBitMasks)
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_INST_MESHLETS_PATH}, {.defines = {{"SetEntityMeshletVisibilityBitMasks_SHADER", "1"}}}},
        .name = std::string{SetEntityMeshletVisibilityBitMasks::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(SetEntityMeshletVisibilityBitMasks::NAME));
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

struct PrepopInfo
{
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView visible_meshlets_prev = {};
    daxa::TaskBufferView meshlet_instances_last_frame = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView entity_meshlet_visibility_bitfield_offsets = {};
    daxa::TaskBufferView entity_meshlet_visibility_bitfield_arena = {};
};
inline void task_prepopulate_instantiated_meshlets(GPUContext *context, daxa::TaskGraph &tg, PrepopInfo info)
{
    // NVIDIA DRIVER BUGS MAKES VKBUFFERFILL IGNORE OFFSET IF() FILL.SIZE + FILL.OFFSET == BUFFER.SIZE). 
    // WORKAROUND BY DOING A BUFFER 
    std::array<ClearRange, 2> clear_ranges = {
        ClearRange{.value = ENT_MESHLET_VIS_OFFSET_UNALLOCATED, .offset = sizeof(daxa_u32), .size = CLEAR_REST},
        ClearRange{.value = 0, .offset = 0, .size = sizeof(daxa_u32)},
    };
    task_multi_clear_buffer(tg, info.entity_meshlet_visibility_bitfield_offsets, clear_ranges);
    task_clear_buffer(tg, info.meshlet_instances, 0, sizeof(daxa_u32vec2));
    task_clear_buffer(tg, info.entity_meshlet_visibility_bitfield_arena, 0);
    auto command_buffer = tg.create_transient_buffer({sizeof(DispatchIndirectStruct), "cb prepopulate_instantiated_meshlets"});
    tg.add_task(PrepopulateInstantiatedMeshletsCommandWriteTask{
        .uses = {
            .u_visible_meshlets_prev = info.visible_meshlets_prev,
            .u_command = command_buffer,
        },
        .context = context,
    });
    tg.add_task(PrepopulateInstantiatedMeshletsTask{
        .uses = {
            .u_command = command_buffer,
            .u_visible_meshlets_prev = info.visible_meshlets_prev,
            .u_instantiated_meshlets_prev = info.meshlet_instances_last_frame,
            .u_meshes = info.meshes,
            .u_instantiated_meshlets = info.meshlet_instances,
            .u_entity_meshlet_visibility_bitfield_offsets = info.entity_meshlet_visibility_bitfield_offsets,
        },
        .context = context,
    });
    tg.add_task(SetEntityMeshletVisibilityBitMasksTask{
        .uses = {
            .u_command = command_buffer,
            .u_instantiated_meshlets = info.meshlet_instances,
            .u_entity_meshlet_visibility_bitfield_offsets = info.entity_meshlet_visibility_bitfield_offsets,
            .u_entity_meshlet_visibility_bitfield_arena = info.entity_meshlet_visibility_bitfield_arena,
        },
        .context = context,
    });
}
#endif