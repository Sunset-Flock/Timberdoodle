#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/geometry_pipeline.inl"

#define SFPM_X 128
#define SFPM_ALLOC_ENT_BITFIELD_LISTS_X SFPM_X
#define SFPM_ALLOC_MESH_BITFIELDS_X SFPM_X
#define SFPM_WRITE_MESHLETS_X SFPM_X

// - Goes over all opaque mesh instance draws
// - Allocates a list with an entry for every mesh for each entity
// - This dispatch also writes the indirect command buffer used for the following two passes.
DAXA_DECL_TASK_HEAD_BEGIN(AllocEntBitfieldListsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entity_mesh_groups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMeshGroup), mesh_groups)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, SFPMBitfieldRef, bitfield_arena)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(DispatchIndirectStruct), command)
DAXA_DECL_TASK_HEAD_END

// - Goes over all visible meshlets from last frame
// - Allocates a bitfield for every mesh instance
DAXA_DECL_TASK_HEAD_BEGIN(AllocMeshletInstBitfieldsH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, command)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VisibleMeshletList), visible_meshlets_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), meshlet_instances_prev)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, SFPMBitfieldRef, bitfield_arena)
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
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(MeshletInstancesBufferHead), meshlet_instances)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, SFPMBitfieldRef, bitfield_arena)
DAXA_DECL_TASK_HEAD_END

struct AllocEntBitfieldListsPush
{
    DAXA_TH_BLOB(AllocEntBitfieldListsH, attach)
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

#if defined(__cplusplus)

#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

static constexpr char PRE_POPULATE_MESHLET_INSTANCES_PATH[] =
    "./src/rendering/rasterize_visbuffer/select_first_pass_meshlets.glsl";
static constexpr char PRE_POPULATE_MESHLET_INSTANCES_PATH_HLSL[] =
    "./src/rendering/rasterize_visbuffer/select_first_pass_meshlets.hlsl";

inline auto sfpm_allocate_ent_bitfield_lists() -> daxa::ComputePipelineCompileInfo2
{
    return { 
        .source = daxa::ShaderFile{PRE_POPULATE_MESHLET_INSTANCES_PATH_HLSL}, 
        .entry_point = "entry_alloc_ent_bitfield_lists", 
        .push_constant_size = sizeof(AllocEntBitfieldListsPush),
        .name = "test",
    };
}

inline auto alloc_entity_to_mesh_instances_offsets_pipeline_compile_info() -> daxa::ComputePipelineCompileInfo
{
    return {
        .shader_info = daxa::ShaderCompileInfo{daxa::ShaderFile{PRE_POPULATE_MESHLET_INSTANCES_PATH},
            {.defines = {{"AllocEntToMeshInstOffsetsOffsets_SHADER", "1"}}}},
        .push_constant_size =
            s_cast<u32>(sizeof(AllocEntBitfieldListsPush)),
        .name = std::string{AllocEntBitfieldListsH::NAME},
    };
}
struct AllocEntToMeshInstOffsetsOffsetsTask : AllocEntBitfieldListsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        u32 const draw_list_total_count =
            render_context->mesh_instance_counts.prepass_instance_counts[0] +
            render_context->mesh_instance_counts.prepass_instance_counts[1];
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(sfpm_allocate_ent_bitfield_lists().name));
        AllocEntBitfieldListsPush push = {
            .attach = ti.attachment_shader_blob,
        };
        ti.recorder.push_constant(push);

        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_0);
        ti.recorder.dispatch({round_up_div(draw_list_total_count, SFPM_ALLOC_ENT_BITFIELD_LISTS_X), 1, 1});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_0);
    }
};

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
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(prepopulate_meshlet_instances_pipeline_compile_info().name));
        AllocMeshletInstBitfieldsPush push = {
            .attach = ti.attachment_shader_blob,
        };
        ti.recorder.push_constant(push);
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_1);

        if (render_context->render_data.settings.enable_visbuffer_two_pass_culling != 0)
        {
            ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(AllocMeshletInstBitfieldsH::AT.command).ids[0]});
        }
        else
        {
            u32 const draw_list_total_count =
                render_context->mesh_instance_counts.prepass_instance_counts[0] +
                render_context->mesh_instance_counts.prepass_instance_counts[1];
            ti.recorder.dispatch({round_up_div(draw_list_total_count, SFPM_ALLOC_ENT_BITFIELD_LISTS_X), 1, 1});
        }
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_1);
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
    RenderContext * render_context = {};
    void callback(daxa::TaskInterface ti)
    {
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(set_entity_meshlets_visibility_bitmasks_pipeline_compile_info().name));
        WriteFirstPassMeshletsAndBitfieldsPush push = {
            .attach = ti.attachment_shader_blob,
        };
        ti.recorder.push_constant(push);
        render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_FIRST_PASS_SELECT_MESHLETS);
        ti.recorder.dispatch_indirect({.indirect_buffer = ti.get(WriteFirstPassMeshletsAndBitfieldsH::AT.command).ids[0]});
        render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::VISBUFFER_FIRST_PASS_SELECT_MESHLETS);
    }
};

struct SelectFirstPassMeshletsInfo
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
    daxa::TaskBufferView & first_pass_meshlets_bitfield_arena;
};
inline void task_select_first_pass_meshlets(SelectFirstPassMeshletsInfo info)
{
    // Process of selecting meshlets for the first pass:
    // - Build indirect bitfield, each bit noting if a meshlet is to be drawn in the first pass
    // - the bitfield is indirect, shaders have to provide the entity, meshgroup and meshlst index to read a bit
    // - its build in four passes:
    //     1. clear bitfield arena to zero. (its very small so the full clear is fast)
    //     2. allocate indirections from entity to mesh offset list
    //     3. allocate indirections for each mesh in the mesh offset lists into the bitfield
    //     4. set bits in bitfield AND write list of meshlet instances

    // - for sep 2 a dispatch goes over all opaque mesh instances, a single thread is elected to allocate the entities offset.
    //     - when the entity is not drawn this frame, its offset stays 0
    // - for step 3 and 4, the dispatches go over all visible meshlets from the last frame.
    //     - in step 3, one thread is elected for each mesh group index to allocate the bitfield for that mesh
    //     - in step 5, each thread sets the bit for its meshlet. They also each write the meshlet to the new frames meshlet instance list.

    // Memory layout:
    // INDEX 0                to MAX_ENTITIES contain entity to mesh list offsets
    // INDEX MAX_ENTITIES                     contains a dynamic allocation offset
    // INDEX MAX_ENTITIES+1   to N            contains meshlists
    // INDEX N                to N1           contains bitfields
    auto first_pass_meshlets_bitfield_arena = info.tg.create_transient_buffer({
        .size = sizeof(daxa_u32) * FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE,
        .name = "first pass meshlet bitfield arena",
    });
    info.first_pass_meshlets_bitfield_arena = first_pass_meshlets_bitfield_arena;

    info.tg.clear_buffer({.buffer = first_pass_meshlets_bitfield_arena, .clear_value = 0});

    auto command_buffer = info.tg.create_transient_buffer({
        sizeof(DispatchIndirectStruct),
        "cb prepopulate_meshlet_instances",
    });

    info.tg.add_task(AllocEntToMeshInstOffsetsOffsetsTask{
        .views = std::array{
            daxa::attachment_view(AllocEntBitfieldListsH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(AllocEntBitfieldListsH::AT.mesh_instances, info.mesh_instances),
            daxa::attachment_view(AllocEntBitfieldListsH::AT.entity_mesh_groups, info.entity_mesh_groups),
            daxa::attachment_view(AllocEntBitfieldListsH::AT.mesh_groups, info.mesh_group_manifest),
            daxa::attachment_view(AllocEntBitfieldListsH::AT.bitfield_arena, first_pass_meshlets_bitfield_arena),
            daxa::attachment_view(AllocEntBitfieldListsH::AT.visible_meshlets_prev, info.visible_meshlets_prev),
            daxa::attachment_view(AllocEntBitfieldListsH::AT.command, command_buffer),
        },
        .render_context = info.render_context,
    });

    info.tg.add_task(AllocMeshletInstBitfieldsTask{
        .views = std::array{
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.command, command_buffer),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.mesh_instances, info.mesh_instances),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.visible_meshlets_prev, info.visible_meshlets_prev),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.meshlet_instances_prev, info.meshlet_instances_last_frame),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.meshes, info.meshes),
            daxa::attachment_view(AllocMeshletInstBitfieldsH::AT.bitfield_arena, first_pass_meshlets_bitfield_arena),
        },
        .render_context = info.render_context,
    });

    if (info.render_context->render_data.settings.enable_visbuffer_two_pass_culling != 0)
    {
        info.tg.add_task(WriteFirstPassMeshletsAndBitfieldsTask{
            .views = std::array{
                daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.globals, info.render_context->tgpu_render_data),
                daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.command, command_buffer),
                daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.visible_meshlets_prev, info.visible_meshlets_prev),
                daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.meshlet_instances_prev, info.meshlet_instances_last_frame),
                daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.materials, info.materials),
                daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.meshlet_instances, info.meshlet_instances),
                daxa::attachment_view(WriteFirstPassMeshletsAndBitfieldsH::AT.bitfield_arena, first_pass_meshlets_bitfield_arena),
            },
            .render_context = info.render_context,
        });
    }
}
#endif