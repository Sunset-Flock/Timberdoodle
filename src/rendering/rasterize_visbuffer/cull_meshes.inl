#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/scene.inl"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/gpu_work_expansion.inl"

#define CULL_MESHES_WORKGROUP_X 128

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(ExpandMeshesToMeshletsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, hiz)       // OPTIONAL
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, hip) // OPTIONAL
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D_ARRAY, point_hip) // OPTIONAL
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(uint), opaque_expansion)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(uint), masked_expansion)
// TODO REMOVE, PUT IN VSM GLOBALS
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_DECL_TASK_HEAD_END

struct ExpandMeshesToMeshletsAttachments
{
    ExpandMeshesToMeshletsH::AttachmentShaderBlob attachments;
}
DAXA_DECL_BUFFER_PTR(ExpandMeshesToMeshletsAttachments);

struct ExpandMeshesToMeshletsPush
{
    daxa_BufferPtr(ExpandMeshesToMeshletsAttachments) attachments;
    daxa::b32 cull_meshes;
    daxa::b32 cull_against_last_frame; /// WARNING: only supported for non vsm path!
    daxa_BufferPtr(GPUMesh) meshes;
    daxa_BufferPtr(daxa_f32mat4x3) entity_combined_transforms;
    daxa::i32 mip_level;
};

#if defined(__cplusplus)
#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

static constexpr inline char const CULL_MESHES_SHADER_PATH[] = "./src/rendering/rasterize_visbuffer/cull_meshes.hlsl";

inline MAKE_COMPUTE_COMPILE_INFO(expand_meshes_pipeline_compile_info, "./src/rendering/rasterize_visbuffer/cull_meshes.hlsl", "main")

    struct ExpandMeshesToMeshletsTask : ExpandMeshesToMeshletsH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};
    bool cull_meshes = {};
    bool cull_against_last_frame = {};
    u32 render_time_index = ~0u;
    bool is_point_light = {};
    bool is_directional_light = {};
    i32 mip_level = {};

    void callback(daxa::TaskInterface ti)
    {
        DBG_ASSERT_TRUE_M(!(is_point_light && is_directional_light), "Cannot be both directional and point light");
        ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(expand_meshes_pipeline_compile_info().name));

        auto alloc = ti.allocator->allocate(sizeof(ExpandMeshesToMeshletsAttachments));
        std::memcpy(alloc->host_address, ti.attachment_shader_blob.data(), sizeof(ExpandMeshesToMeshletsH::AttachmentShaderBlob));
        ExpandMeshesToMeshletsPush push = {
            .attachments = alloc->device_address,
            .cull_meshes = cull_meshes,
            .cull_against_last_frame = cull_against_last_frame,
            .meshes = render_context->render_data.scene.meshes,
            .entity_combined_transforms = render_context->render_data.scene.entity_combined_transforms,
            .mip_level = mip_level,
        };
        ti.recorder.push_constant(push);
        auto total_mesh_draws =
            render_context->mesh_instance_counts.prepass_instance_counts[0] +
            render_context->mesh_instance_counts.prepass_instance_counts[1];
        total_mesh_draws = std::min(total_mesh_draws, MAX_MESH_INSTANCES);

        daxa::DispatchInfo dispatch_info = {round_up_div(total_mesh_draws, CULL_MESHES_WORKGROUP_X), 1, 1};
        if(is_point_light)
        {
            dispatch_info.y = 6; // Mip count
            dispatch_info.z = render_context->render_data.vsm_settings.point_light_count;
        }
        else if (is_directional_light) 
        {
            dispatch_info.y = VSM_CLIP_LEVELS;
        }

        render_context->render_times.start_gpu_timer(ti.recorder, render_time_index);
        ti.recorder.dispatch(dispatch_info);
        render_context->render_times.end_gpu_timer(ti.recorder, render_time_index);
    }
};

struct TaskExpandMeshesToMeshletsInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph & tg;
    bool cull_meshes = {};
    bool cull_against_last_frame = {};
    u32 render_time_index = RenderTimes::INVALID_RENDER_TIME_INDEX;
    // Used for VSM page culling:
    daxa::TaskImageView vsm_hip = daxa::NullTaskImage;
    // Used for VSM point page culling:
    daxa::TaskImageView vsm_point_hip = daxa::NullTaskImage;
    bool is_point_light = false;
    bool is_directional_light = false;
    daxa::i32 mip_level = {-1};
    daxa::TaskBufferView vsm_clip_projections = daxa::NullTaskBuffer;
    daxa::TaskBufferView vsm_point_lights = daxa::NullTaskBuffer;
    daxa::TaskBufferView vsm_globals = daxa::NullTaskBuffer;
    daxa::TaskImageView hiz = daxa::NullTaskImage;
    daxa::TaskBufferView globals = {};
    daxa::TaskBufferView mesh_instances = {};
    std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> & meshlet_expansions;
    DispatchIndirectStruct dispatch_clear = {0, 1, 1};
    std::string buffer_name_prefix = "";
};
void tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo const & info)
{
    bool const prefix_sum_expansion = info.render_context->render_data.settings.enable_prefix_sum_work_expansion;

    auto const expansion_size = prefix_sum_expansion ? PrefixSumWorkExpansionBufferHead::calc_buffer_size(MAX_MESH_INSTANCES) : Po2PackedWorkExpansionBufferHead::calc_buffer_size(MAX_MESH_INSTANCES);
    auto opaque_expansion = info.tg.create_transient_buffer({
        .size = expansion_size,
        .name = info.buffer_name_prefix + "opaque_meshlet_expansion_buffer" + std::to_string(rand()),
    });
    auto masked_expansion = info.tg.create_transient_buffer({
        .size = expansion_size,
        .name = info.buffer_name_prefix + "masked_meshlet_expansion_buffer" + std::to_string(rand()),
    });
    info.tg.add_task(daxa::InlineTask{std::string("clear work expansion buffer") + std::to_string(rand())}
            .tf.writes(opaque_expansion, masked_expansion)
            .executes(
                [=](daxa::TaskInterface ti)
                {
                    if (prefix_sum_expansion)
                    {
                        allocate_fill_copy(
                            ti, PrefixSumWorkExpansionBufferHead::create(ti.device_address(opaque_expansion).value(), MAX_MESH_INSTANCES, info.dispatch_clear),
                            ti.get(opaque_expansion));
                        allocate_fill_copy(
                            ti, PrefixSumWorkExpansionBufferHead::create(ti.device_address(masked_expansion).value(), MAX_MESH_INSTANCES, info.dispatch_clear),
                            ti.get(masked_expansion));
                    }
                    else
                    {
                        allocate_fill_copy(
                            ti, Po2PackedWorkExpansionBufferHead::create(ti.device_address(opaque_expansion).value(), MAX_MESH_INSTANCES, info.dispatch_clear),
                            ti.get(opaque_expansion));
                        allocate_fill_copy(
                            ti, Po2PackedWorkExpansionBufferHead::create(ti.device_address(masked_expansion).value(), MAX_MESH_INSTANCES, info.dispatch_clear),
                            ti.get(masked_expansion));
                    }
                }));
    info.meshlet_expansions = std::array{opaque_expansion, masked_expansion};

    info.tg.add_task(ExpandMeshesToMeshletsTask{
        .views = ExpandMeshesToMeshletsTask::Views{
            .globals = info.globals,
            .mesh_instances = info.mesh_instances,
            .hiz = info.hiz,
            .hip = info.vsm_hip,
            .point_hip = info.vsm_point_hip,
            .opaque_expansion = info.meshlet_expansions[0],
            .masked_expansion = info.meshlet_expansions[1],
            .vsm_clip_projections = info.vsm_clip_projections,
            .vsm_point_lights = info.vsm_point_lights,
            .vsm_globals = info.vsm_globals,
        },
        .render_context = info.render_context,
        .cull_meshes = info.cull_meshes,
        .cull_against_last_frame = info.cull_against_last_frame,
        .render_time_index = info.render_time_index,
        .is_point_light = info.is_point_light,
        .is_directional_light = info.is_directional_light,
        .mip_level = info.mip_level
    });
}

#endif