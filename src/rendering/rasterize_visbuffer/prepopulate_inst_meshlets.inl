#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/asset.inl"
#define PREPOPULATE_INST_MESHLETS_X 256

DAXA_DECL_TASK_HEAD_BEGIN(PrepopInstMeshletCommW, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), command)
DAXA_DECL_TASK_HEAD_END

// In the future we should check if the entity slot is actually valid here.
// To do that we need a version in the entity id and a version table we can compare to
DAXA_DECL_TASK_HEAD_BEGIN(PrepopulateInstMeshlets, 6)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(SetEntityMeshletVisibilityBitMasks, 4)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(DispatchIndirectStruct), command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstances), instantiated_meshlets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, EntityMeshletVisibilityBitfieldOffsetsView, entity_meshlet_visibility_bitfield_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), entity_meshlet_visibility_bitfield_arena)
DAXA_DECL_TASK_HEAD_END

struct PrepopInstMeshletCommWPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DAXA_TH_BLOB(PrepopInstMeshletCommW, uses)
};

struct PrepopulateInstMeshletsPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DAXA_TH_BLOB(PrepopulateInstMeshlets, uses)
};

struct SetEntityMeshletVisibilityBitMasksPush
{
    daxa_BufferPtr(ShaderGlobals) globals;
    DAXA_TH_BLOB(SetEntityMeshletVisibilityBitMasks, uses)
};

#if __cplusplus

#include "../../gpu_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const PRE_POPULATE_INST_MESHLETS_PATH[] =
    "./src/rendering/rasterize_visbuffer/prepopulate_inst_meshlets.glsl";

using PrepopulateInstantiatedMeshletsCommandWriteTask =
    WriteIndirectDispatchArgsPushBaseTask<PrepopInstMeshletCommW, PRE_POPULATE_INST_MESHLETS_PATH, PrepopInstMeshletCommWPush>;
auto prepopulate_instantiated_meshlets_command_write_pipeline_compile_info()
{
    return write_indirect_dispatch_args_base_compile_pipeline_info<
        PrepopInstMeshletCommW, PRE_POPULATE_INST_MESHLETS_PATH, PrepopInstMeshletCommWPush>();
};

inline daxa::ComputePipelineCompileInfo prepopulate_inst_meshlets_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_INST_MESHLETS_PATH}},
        .push_constant_size = s_cast<u32>(sizeof(PrepopulateInstMeshletsPush) + PrepopulateInstMeshlets::attachment_shader_data_size()),
        .name = std::string{PrepopulateInstMeshlets{}.name()},
    };
}
struct PrepopulateInstantiatedMeshletsTask : PrepopulateInstMeshlets
{
    PrepopulateInstMeshlets::AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(PrepopulateInstMeshlets{}.name()));
        ti.recorder.push_constant(PrepopulateInstMeshletsPush{.globals = context->shader_globals_address});
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
            .offset = sizeof(PrepopulateInstMeshletsPush),
        });
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(PrepopulateInstMeshlets::command).ids[0]});
    }
};

inline daxa::ComputePipelineCompileInfo set_entity_meshlets_visibility_bitmasks_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_INST_MESHLETS_PATH},
            {.defines = {{"SetEntityMeshletVisibilityBitMasks_SHADER", "1"}}}},
        .push_constant_size =
            s_cast<u32>(sizeof(SetEntityMeshletVisibilityBitMasksPush) +
                        SetEntityMeshletVisibilityBitMasks::attachment_shader_data_size()),
        .name = std::string{SetEntityMeshletVisibilityBitMasks{}.name()},
    };
}
struct SetEntityMeshletVisibilityBitMasksTask : SetEntityMeshletVisibilityBitMasks
{
    SetEntityMeshletVisibilityBitMasks::AttachmentViews views = {};
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*context->compute_pipelines.at(SetEntityMeshletVisibilityBitMasks{}.name()));
        ti.recorder.push_constant(SetEntityMeshletVisibilityBitMasksPush{.globals = context->shader_globals_address});
        ti.recorder.push_constant_vptr({
            .data = ti.attachment_shader_data.data(),
            .size = ti.attachment_shader_data.size(),
            .offset = sizeof(SetEntityMeshletVisibilityBitMasksPush),
        });
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(SetEntityMeshletVisibilityBitMasks::command).ids[0]});
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
inline void task_prepopulate_instantiated_meshlets(GPUContext * context, daxa::TaskGraph & tg, PrepopInfo info)
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
    auto command_buffer = tg.create_transient_buffer({
        sizeof(DispatchIndirectStruct),
        "cb prepopulate_instantiated_meshlets",
    });

    tg.add_task(PrepopulateInstantiatedMeshletsCommandWriteTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsCommandWriteTask::visible_meshlets_prev, info.visible_meshlets_prev}},
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsCommandWriteTask::command, command_buffer}},
        },
        .context = context,
    });

    tg.add_task(PrepopulateInstantiatedMeshletsTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsTask::command, command_buffer}},
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsTask::visible_meshlets_prev, info.visible_meshlets_prev}},
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsTask::instantiated_meshlets_prev, info.meshlet_instances_last_frame}},
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsTask::meshes, info.meshes}},
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsTask::instantiated_meshlets, info.meshlet_instances}},
            daxa::TaskViewVariant{std::pair{PrepopulateInstantiatedMeshletsTask::entity_meshlet_visibility_bitfield_offsets, info.entity_meshlet_visibility_bitfield_offsets}},
        },
        .context = context});

    tg.add_task(SetEntityMeshletVisibilityBitMasksTask{
        .views = std::array{
            daxa::TaskViewVariant{std::pair{SetEntityMeshletVisibilityBitMasksTask::command, command_buffer}},
            daxa::TaskViewVariant{std::pair{SetEntityMeshletVisibilityBitMasksTask::instantiated_meshlets, info.meshlet_instances}},
            daxa::TaskViewVariant{std::pair{SetEntityMeshletVisibilityBitMasksTask::entity_meshlet_visibility_bitfield_offsets, info.entity_meshlet_visibility_bitfield_offsets}},
            daxa::TaskViewVariant{std::pair{SetEntityMeshletVisibilityBitMasksTask::entity_meshlet_visibility_bitfield_arena, info.entity_meshlet_visibility_bitfield_arena}},
        },
        .context = context});
}
#endif