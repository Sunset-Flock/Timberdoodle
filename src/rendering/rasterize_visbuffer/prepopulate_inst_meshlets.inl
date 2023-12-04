#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"
#define PREPOPULATE_INST_MESHLETS_X 256

DAXA_DECL_TASK_HEAD_BEGIN(PrepopInstMeshletCommW)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), command)
DAXA_DECL_TASK_HEAD_END

// In the future we should check if the entity slot is actually valid here.
// To do that we need a version in the entity id and a version table we can compare to
DAXA_DECL_TASK_HEAD_BEGIN(PrepopulateInstMeshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(SetEntityMeshletVisibilityBitMasks)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), entity_meshlet_visibility_bitfield_arena)
DAXA_DECL_TASK_HEAD_END

struct PrepopInstMeshletCommWPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    PrepopInstMeshletCommW uses;
};

struct PrepopulateInstMeshletsPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    PrepopulateInstMeshlets uses;
};

struct SetEntityMeshletVisibilityBitMasksPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    SetEntityMeshletVisibilityBitMasks uses;
};

#if __cplusplus

#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const PRE_POPULATE_INST_MESHLETS_PATH[] =
    "./src/rendering/rasterize_visbuffer/prepopulate_inst_meshlets.glsl";

using PrepopulateInstantiatedMeshletsCommandWriteTask = WriteIndirectDispatchArgsPushBaseTask<
    PrepopInstMeshletCommW,
    PRE_POPULATE_INST_MESHLETS_PATH,
    PrepopInstMeshletCommWPush>;

struct PrepopulateInstantiatedMeshletsTask
{
    USE_TASK_HEAD(PrepopulateInstMeshlets)
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_INST_MESHLETS_PATH}},
        .name = std::string{PrepopulateInstMeshlets::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        PrepopulateInstMeshletsPush push = { .globals = context->shader_globals_address };
        ti.copy_task_head_to(&push.uses);
        cmd.set_pipeline(*context->compute_pipelines.at(PrepopulateInstMeshlets::NAME));
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.command.buffer(),
        });
    }
};

struct SetEntityMeshletVisibilityBitMasksTask
{
    USE_TASK_HEAD(SetEntityMeshletVisibilityBitMasks)
    inline static const daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO{
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_INST_MESHLETS_PATH}, {.defines = {{"SetEntityMeshletVisibilityBitMasks_SHADER", "1"}}}},
        .name = std::string{SetEntityMeshletVisibilityBitMasks::NAME},
    };
    GPUContext *context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto & cmd = ti.get_recorder();
        SetEntityMeshletVisibilityBitMasksPush push = { .globals = context->shader_globals_address  };
        ti.copy_task_head_to(&push.uses);
        cmd.set_pipeline(*context->compute_pipelines.at(SetEntityMeshletVisibilityBitMasks::NAME));
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.command.buffer(),
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
            .visible_meshlets_prev = info.visible_meshlets_prev,
            .command = command_buffer,
        },
        .context = context,
    });
    tg.add_task(PrepopulateInstantiatedMeshletsTask{
        .uses = {
            .command = command_buffer,
            .visible_meshlets_prev = info.visible_meshlets_prev,
            .instantiated_meshlets_prev = info.meshlet_instances_last_frame,
            .meshes = info.meshes,
            .instantiated_meshlets = info.meshlet_instances,
            .entity_meshlet_visibility_bitfield_offsets = info.entity_meshlet_visibility_bitfield_offsets,
        },
        .context = context,
    });
    tg.add_task(SetEntityMeshletVisibilityBitMasksTask{
        .uses = {
            .command = command_buffer,
            .instantiated_meshlets = info.meshlet_instances,
            .entity_meshlet_visibility_bitfield_offsets = info.entity_meshlet_visibility_bitfield_offsets,
            .entity_meshlet_visibility_bitfield_arena = info.entity_meshlet_visibility_bitfield_arena,
        },
        .context = context,
    });
}
#endif