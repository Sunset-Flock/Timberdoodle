#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../tasks/memset.inl"

#define ALLOC_ENT_TO_MESH_INST_OFFSETS_OFFSETS_X 128
#define PREPOPULATE_MESHLET_INSTANCES_X 256

DAXA_DECL_TASK_HEAD_BEGIN(AllocEntToMeshInstOffsetsOffsetsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_mesh_groups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), mesh_groups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IndirectMemsetBufferCommand), clear_arena_command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ent_to_mesh_inst_offsets_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, U32ArenaBufferRef, bitfield_arena)
DAXA_DECL_TASK_HEAD_END

DAXA_DECL_TASK_HEAD_BEGIN(PrepopMeshletInstancesCommWH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), command)
DAXA_DECL_TASK_HEAD_END

// - Goes over all visible meshlets from last frame
// - Attempts to allocate a meshlet instance bitfield offset for each mesh in the list of visible meshlets
DAXA_DECL_TASK_HEAD_BEGIN(AllocMeshletInstBitfieldsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ent_to_mesh_inst_offsets_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IndirectMemsetBufferCommand), clear_arena_command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, U32ArenaBufferRef, bitfield_arena)
DAXA_DECL_TASK_HEAD_END

// - Goes over all visible meshlets from last frame again
// - Sets bits for all previously visible meshlets
// - prepopulates meshlet instances with previously visible meshlets
DAXA_DECL_TASK_HEAD_BEGIN(WriteFirstPassMeshletsAndBitfieldsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMaterial), materials)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ent_to_mesh_inst_offsets_offsets)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, U32ArenaBufferRef, bitfield_arena)
DAXA_DECL_TASK_HEAD_END

struct AllocEntToMeshInstOffsetsOffsetsPush
{
    DAXA_TH_BLOB(AllocEntToMeshInstOffsetsOffsetsH, attach)
    daxa_u32 dummy;
};

struct PrepopMeshletInstancesCommWPush
{
    DAXA_TH_BLOB(PrepopMeshletInstancesCommWH, attach)
    daxa_u32 dummy;
};

struct AllocMeshletInstBitfieldsPush
{
    DAXA_TH_BLOB(AllocMeshletInstBitfieldsH, attach)
    daxa_u32 dummy;
};

struct WriteFirstPassMeshletsAndBitfieldsPush
{
    DAXA_TH_BLOB(WriteFirstPassMeshletsAndBitfieldsH, attach)
    daxa_u32 dummy;
};

#if __cplusplus

#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

static constexpr char PRE_POPULATE_MESHLET_INSTANCES_PATH[] =
    "./src/rendering/rasterize_visbuffer/prepopulate_meshlets.glsl";

using AllocMeshletInstBitfieldsCommandWriteTask = SimpleComputeTask<
    PrepopMeshletInstancesCommWH::Task,
    PrepopMeshletInstancesCommWPush,
    PRE_POPULATE_MESHLET_INSTANCES_PATH,
    "main">;

// using AllocMeshletInstBitfieldsCommandWriteTask =
//     WriteIndirectDispatchArgsPushBaseTask<
//     PrepopMeshletInstancesCommW,
//     PRE_POPULATE_MESHLET_INSTANCES_PATH,
//     PrepopMeshletInstancesCommWPush>;

inline auto prepopulate_meshlet_instances_pipeline_compile_info() -> daxa::ComputePipelineCompileInfo
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_MESHLET_INSTANCES_PATH},
            {.defines = {{"AllocMeshletInstBitfields_SHADER", "1"}}}},
        .push_constant_size = s_cast<u32>(sizeof(AllocMeshletInstBitfieldsPush)),
        .name = std::string{AllocMeshletInstBitfieldsH::NAME},
    };
}
struct AllocMeshletInstBitfieldsTask : AllocMeshletInstBitfieldsH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(prepopulate_meshlet_instances_pipeline_compile_info().name));
        AllocMeshletInstBitfieldsPush push = {
            .attach = ti.attachment_shader_blob,
        };
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(AllocMeshletInstBitfieldsH::AT.command).ids[0]});
    }
};

inline auto set_entity_meshlets_visibility_bitmasks_pipeline_compile_info() -> daxa::ComputePipelineCompileInfo
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_MESHLET_INSTANCES_PATH},
            {.defines = {{"WriteFirstPassMeshletsAndBitfields_SHADER", "1"}}}},
        .push_constant_size =
            s_cast<u32>(sizeof(WriteFirstPassMeshletsAndBitfieldsPush)),
        .name = std::string{WriteFirstPassMeshletsAndBitfieldsH::NAME},
    };
}
struct WriteFirstPassMeshletsAndBitfieldsTask : WriteFirstPassMeshletsAndBitfieldsH::Task
{
    AttachmentViews views = {};
    GPUContext * gpu_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*gpu_context->compute_pipelines.at(set_entity_meshlets_visibility_bitmasks_pipeline_compile_info().name));
        WriteFirstPassMeshletsAndBitfieldsPush push = {
            .attach = ti.attachment_shader_blob,
        };
        ti.recorder.push_constant(push);
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(WriteFirstPassMeshletsAndBitfieldsH::AT.command).ids[0]});
    }
};

inline auto alloc_entity_to_mesh_instances_offsets_pipeline_compile_info() -> daxa::ComputePipelineCompileInfo
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_MESHLET_INSTANCES_PATH},
            {.defines = {{"AllocEntToMeshInstOffsetsOffsets_SHADER", "1"}}}},
        .push_constant_size =
            s_cast<u32>(sizeof(AllocEntToMeshInstOffsetsOffsetsPush)),
        .name = std::string{AllocEntToMeshInstOffsetsOffsetsH::NAME},
    };
}
struct AllocEntToMeshInstOffsetsOffsetsTask : AllocEntToMeshInstOffsetsOffsetsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        u32 const draw_list_total_count =
            render_context->mesh_instance_counts.prepass_instance_counts[0] +
            render_context->mesh_instance_counts.prepass_instance_counts[1];
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(alloc_entity_to_mesh_instances_offsets_pipeline_compile_info().name));
        AllocEntToMeshInstOffsetsOffsetsPush push = {
            .attach = ti.attachment_shader_blob,
        };
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({round_up_div(draw_list_total_count, ALLOC_ENT_TO_MESH_INST_OFFSETS_OFFSETS_X),
            1,
            1});
    }
};

struct PrepopInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & tg;
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskBufferView meshes = {};
    daxa::TaskBufferView materials = {};
    daxa::TaskBufferView entity_mesh_groups = {};
    daxa::TaskBufferView mesh_group_manifest = {};
    daxa::TaskBufferView visible_meshlets_prev = {};
    daxa::TaskBufferView meshlet_instances_last_frame = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView & first_pass_meshlets_bitfield_offsets;
    daxa::TaskBufferView & first_pass_meshlets_bitfield_arena;
};
inline void task_prepopulate_meshlet_instances(PrepopInfo info)
{
    // MeshInstance:
    // - Each entity has a list of meshes, its meshgroup.
    // - A mesh paired with an entity is a mesh instance.
    // - So each mesh in an entities meshgroup is a mesh instance.

    // Prepopulate Meshlet Instances:
    // - Fills MeshInstance lists for first pass.
    // - Fills Bitfield, each bit describing if a meshet of the draw lists was drawn in first pass
    //   - U32Arena: a buffer used for quick, per-frame linear allocation containing bitfields and offsets
    //   - entity_to_mesh_group_offsets_offsets:
    //       buffer containing array with an entry per entity.
    //       each entry is an offset into a list of offsets.
    //       each of these offsets points to a section in the arena.
    //       each of these sections is a bitfield for a mesh of the mesh group of that entity
    // - Bitfields are used in second pass to skip already drawn meshlets.

    // Process:
    // - clear entity_to_mesh_group_offsets_offsets
    // - clear the two counters within the arena
    // - allocate entity meshgroup offsets offsets: for each opaque draw:
    //   - allocate offset for mesh bitfield offsets for each entity
    //   - write indirect command for following clear
    // - indirect clear section of arena used by mesh offsets to INVALID_ENTITY_TO_MESHGROUP_BITFIELD_OFFSET_OFFSET with 0 WITH OFFSET 8!!!
    // - prepopulate meshlets: for each visible meshlet last frame:
    //   - try to allocate bitfield for each mesh in each meshgroup for each entity
    //     - if out of memory, do NOT insert meshlet into meshlet instance list
    //     - if allocation success:
    //       - write out meshlet into meshlet instance list
    //       - write out meshlet instance index to opaque draw list
    //       - write the allocated offset to the mesh instance's offset
    //   - write command for following clear
    // - indirect clear U32ArenaBufferRef from offsets_section_size to bitfield_section_size with 0 WITH OFFSET 8!!!
    // - Set first pass meshlet bits: for each meshlet instance:
    //   - atomically or meshlet bit on bitfield entry

    // Each mesh instance gets a bitfield that describes which meshlets of that mesh instance were drawin in pass 0.
    // AllocateEntityToMeshInstanceBitfieldOffsets allocates a list of offsets for each entity.
    // Each entities list has entries for each mesh instance of that entity.
    // THe entries are offsets into the arena, pointing to that mesh instances bitfield.
    auto ent_to_mesh_inst_offsets_offsets = info.tg.create_transient_buffer({
        .size = sizeof(daxa_u32) * MAX_ENTITIES,
        .name = "entity_to_mesh_group_offsets_buffer",
    });
    auto first_pass_meshlets_bitfield_arena = info.tg.create_transient_buffer({
        .size = sizeof(daxa_u32) * FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE,
        .name = "first pass meshlet bitfield arena",
    });
    info.first_pass_meshlets_bitfield_offsets = ent_to_mesh_inst_offsets_offsets;
    info.first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena;

    // TODO: only clear values for entities that are present in the mesh instances list.
    info.tg.clear_buffer({.buffer = ent_to_mesh_inst_offsets_offsets, .clear_value = FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID });

    // TODO: only clear used part of buffer.
    info.tg.clear_buffer({.buffer = first_pass_meshlets_bitfield_arena, .clear_value = 0});

    auto clear_mesh_instance_bitfield_offsets_command = info.tg.create_transient_buffer({sizeof(IndirectMemsetBufferCommand), "clear_mesh_instance_bitfield_offsets_command"});
    task_fill_buffer(
        info.tg, clear_mesh_instance_bitfield_offsets_command,
        IndirectMemsetBufferCommand{
            .dispatch = {0, 1, 1},
            .offset = 2,
            .size = 0,
            .value = FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID,
        });

    info.tg.add_task(AllocEntToMeshInstOffsetsOffsetsTask{
        .views = std::array{
            daxa::attachment_view(AllocEntToMeshInstOffsetsOffsetsH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(AllocEntToMeshInstOffsetsOffsetsH::AT.mesh_instances, info.mesh_instances),
            daxa::attachment_view(AllocEntToMeshInstOffsetsOffsetsH::AT.entity_mesh_groups, info.entity_mesh_groups),
            daxa::attachment_view(AllocEntToMeshInstOffsetsOffsetsH::AT.mesh_groups, info.mesh_group_manifest),
            daxa::attachment_view(AllocEntToMeshInstOffsetsOffsetsH::AT.clear_arena_command, clear_mesh_instance_bitfield_offsets_command),
            daxa::attachment_view(AllocEntToMeshInstOffsetsOffsetsH::AT.ent_to_mesh_inst_offsets_offsets, ent_to_mesh_inst_offsets_offsets),
            daxa::attachment_view(AllocEntToMeshInstOffsetsOffsetsH::AT.bitfield_arena, first_pass_meshlets_bitfield_arena),
        },
        .render_context = info.render_context,
    });

    info.tg.add_task(IndirectMemsetBufferTask{
        .views = std::array{
            daxa::attachment_view(IndirectMemsetBufferH::AT.command, clear_mesh_instance_bitfield_offsets_command),
            daxa::attachment_view(IndirectMemsetBufferH::AT.dst, first_pass_meshlets_bitfield_arena),
        },
        .gpu_context = info.render_context->gpu_context,
    });

    auto clear_bitfields_command = info.tg.create_transient_buffer({sizeof(IndirectMemsetBufferCommand), "clear_bitfields_command"});
    task_fill_buffer(info.tg, clear_bitfields_command, IndirectMemsetBufferCommand{
                                                           .dispatch = {0, 1, 1},
                                                           .offset = 0,
                                                           .size = 0,
                                                           .value = 0,
                                                       });

    auto command_buffer = info.tg.create_transient_buffer({
        sizeof(DispatchIndirectStruct),
        "cb prepopulate_meshlet_instances",
    });

    info.tg.add_task(AllocMeshletInstBitfieldsCommandWriteTask{
        .views = std::array{
            daxa::attachment_view(PrepopMeshletInstancesCommWH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(PrepopMeshletInstancesCommWH::AT.visible_meshlets_prev, info.visible_meshlets_prev),
            daxa::attachment_view(PrepopMeshletInstancesCommWH::AT.command, command_buffer),
        },
        .gpu_context = info.render_context->gpu_context,
    });

    info.tg.add_task(AllocMeshletInstBitfieldsTask{
        .views = std::array{
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.command, command_buffer),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.visible_meshlets_prev, info.visible_meshlets_prev),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.meshlet_instances_prev, info.meshlet_instances_last_frame),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.meshes, info.meshes),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.ent_to_mesh_inst_offsets_offsets, ent_to_mesh_inst_offsets_offsets),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.clear_arena_command, clear_bitfields_command),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.bitfield_arena, first_pass_meshlets_bitfield_arena),
        },
        .gpu_context = info.render_context->gpu_context,
    });

    info.tg.add_task(IndirectMemsetBufferTask{
        .views = std::array{
            daxa::attachment_view(IndirectMemsetBufferH::AT.command, clear_bitfields_command),
            daxa::attachment_view(IndirectMemsetBufferH::AT.dst, first_pass_meshlets_bitfield_arena),
        },
        .gpu_context = info.render_context->gpu_context,
    });

    info.tg.add_task(WriteFirstPassMeshletsAndBitfieldsTask{
        .views = std::array{
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.command, command_buffer),
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.visible_meshlets_prev, info.visible_meshlets_prev),
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.meshlet_instances_prev, info.meshlet_instances_last_frame),
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.materials, info.materials),
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.ent_to_mesh_inst_offsets_offsets, ent_to_mesh_inst_offsets_offsets),
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.meshlet_instances, info.meshlet_instances),
            daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.bitfield_arena, first_pass_meshlets_bitfield_arena),
        },
        .gpu_context = info.render_context->gpu_context,
    });
}
#endif