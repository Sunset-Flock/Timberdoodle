#pragma once

#include "cull_meshes.inl"

#include "../scene_renderer_context.hpp"
#include "../tasks/misc.hpp"

inline MAKE_COMPUTE_COMPILE_INFO(expand_meshes_pipeline_compile_info, "./src/rendering/rasterize_visbuffer/cull_meshes.hlsl", "main")

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
    bool is_point_spot_light = false;
    bool is_directional_light = false;
    daxa::i32 mip_level = {-1};
    daxa::TaskBufferView vsm_clip_projections = daxa::NullTaskBuffer;
    daxa::TaskBufferView vsm_point_lights = daxa::NullTaskBuffer;
    daxa::TaskBufferView vsm_spot_lights = daxa::NullTaskBuffer;
    daxa::TaskBufferView vsm_globals = daxa::NullTaskBuffer;
    daxa::TaskImageView hiz = daxa::NullTaskImage;
    daxa::TaskBufferView globals = {};
    daxa::TaskBufferView first_pass_meshlet_bitfield = daxa::NullTaskBuffer;
    daxa::TaskBufferView mesh_instances = {};
    std::array<daxa::TaskBufferView, PREPASS_DRAW_LIST_TYPE_COUNT> & meshlet_expansions;
    DispatchIndirectStruct dispatch_clear = {0, 1, 1};
    std::string_view buffer_name_prefix = "";
};
inline void tasks_expand_meshes_to_meshlets(TaskExpandMeshesToMeshletsInfo const & info)
{
    bool const prefix_sum_expansion = info.render_context->render_data.settings.enable_prefix_sum_work_expansion;

    bool shadow_pass = info.is_directional_light;
    u32 worst_mesh_instances_in_expansion = shadow_pass ? (VSM_CLIP_LEVELS/2) * MAX_MESH_INSTANCES : MAX_MESH_INSTANCES;
    auto const expansion_size = prefix_sum_expansion ? PrefixSumWorkExpansionBufferHead::calc_buffer_size(worst_mesh_instances_in_expansion) : Po2BucketWorkExpansionBufferHead::calc_buffer_size(worst_mesh_instances_in_expansion);
    auto opaque_expansion = info.tg.create_transient_buffer({
        .size = expansion_size,
        .name = std::string(info.buffer_name_prefix) + "opaque_meshlet_expansion_buffer" + std::to_string(rand()),
    });
    auto masked_expansion = info.tg.create_transient_buffer({
        .size = expansion_size,
        .name = std::string(info.buffer_name_prefix) + "masked_meshlet_expansion_buffer" + std::to_string(rand()),
    });
    info.tg.add_task(daxa::InlineTask{std::string("clear work expansion buffer") + std::to_string(rand())}
        .transfer.writes(opaque_expansion, masked_expansion)
        .executes( [=](daxa::TaskInterface ti)
        {
            if (prefix_sum_expansion)
            {
                allocate_fill_copy(
                    ti, PrefixSumWorkExpansionBufferHead::create(ti.device_address(opaque_expansion).value(), worst_mesh_instances_in_expansion, info.dispatch_clear),
                    ti.get(opaque_expansion));
                allocate_fill_copy(
                    ti, PrefixSumWorkExpansionBufferHead::create(ti.device_address(masked_expansion).value(), worst_mesh_instances_in_expansion, info.dispatch_clear),
                    ti.get(masked_expansion));
            }
            else
            {
                allocate_fill_copy(
                    ti, Po2BucketWorkExpansionBufferHead::create(ti.device_address(opaque_expansion).value(), worst_mesh_instances_in_expansion, info.dispatch_clear),
                    ti.get(opaque_expansion));
                allocate_fill_copy(
                    ti, Po2BucketWorkExpansionBufferHead::create(ti.device_address(masked_expansion).value(), worst_mesh_instances_in_expansion, info.dispatch_clear),
                    ti.get(masked_expansion));
            }
        }));
    info.meshlet_expansions = std::array{opaque_expansion, masked_expansion};

    auto info_copy = info;

    info.tg.add_task(
        daxa::HeadTask<ExpandMeshesToMeshletsH::Info>()
            .head_views({
                .globals = info.globals,
                .mesh_instances = info.mesh_instances,
                .hiz = info.hiz,
                .hip = info.vsm_hip,
                .point_hip = info.vsm_point_hip,
                .first_pass_meshlet_bitfield = info.first_pass_meshlet_bitfield,
                .opaque_expansion = info.meshlet_expansions[0],
                .masked_expansion = info.meshlet_expansions[1],
                .vsm_clip_projections = info.vsm_clip_projections,
                .vsm_point_lights = info.vsm_point_lights,
                .vsm_spot_lights = info.vsm_spot_lights,
                .vsm_globals = info.vsm_globals,
            })
            .executes(
                [=](daxa::TaskInterface ti)
                {
                    DBG_ASSERT_TRUE_M(!(info.is_point_spot_light && info.is_directional_light), "Cannot be both directional and point light");
                    ti.recorder.set_pipeline(*info.render_context->gpu_context->compute_pipelines.at(expand_meshes_pipeline_compile_info().name));

                    auto alloc = ti.allocator->allocate(sizeof(ExpandMeshesToMeshletsAttachments));
                    std::memcpy(alloc->host_address, ti.attachment_shader_blob.data(), sizeof(ExpandMeshesToMeshletsH::AttachmentShaderBlob));
                    ExpandMeshesToMeshletsPush push = {
                        .attachments = alloc->device_address,
                        .cull_meshes = info_copy.cull_meshes,
                        .cull_against_last_frame = info_copy.cull_against_last_frame,
                        .meshes = info_copy.render_context->render_data.scene.meshes,
                        .mesh_lod_groups = info_copy.render_context->render_data.scene.mesh_lod_groups,
                        .entity_combined_transforms = info_copy.render_context->render_data.scene.entity_combined_transforms,
                        .mip_level = info_copy.mip_level,
                    };
                    ti.recorder.push_constant(push);
                    auto total_mesh_draws =
                        info_copy.render_context->mesh_instance_counts.prepass_instance_counts[0] +
                        info_copy.render_context->mesh_instance_counts.prepass_instance_counts[1];
                    total_mesh_draws = std::min(total_mesh_draws, MAX_MESH_INSTANCES);

                    daxa::DispatchInfo dispatch_info = {round_up_div(total_mesh_draws, CULL_MESHES_WORKGROUP_X), 1, 1};

                    if (info_copy.is_point_spot_light)
                    {
                        dispatch_info.y =
                            (info_copy.render_context->render_data.vsm_settings.point_light_count * 6) + // 6 -> cube map face count
                            info_copy.render_context->render_data.vsm_settings.spot_light_count;
                    }
                    else if (info_copy.is_directional_light)
                    {
                        dispatch_info.y = VSM_CLIP_LEVELS;
                    }
                    if(info_copy.is_point_spot_light && info_copy.mip_level == 0)
                    {
                        info_copy.render_context->render_times.start_gpu_timer(ti.recorder, RenderTimes::index<"VSM","CULL_MESHES_POINT_SPOT">());
                    }
                    else
                    {
                        info_copy.render_context->render_times.start_gpu_timer(ti.recorder, info_copy.render_time_index);
                    }
                    ti.recorder.dispatch(dispatch_info);
                    if(info_copy.is_point_spot_light && info_copy.mip_level == 6)
                    {
                        info_copy.render_context->render_times.end_gpu_timer(ti.recorder, RenderTimes::index<"VSM","CULL_MESHES_POINT_SPOT">());
                    }
                    else
                    {
                        info_copy.render_context->render_times.end_gpu_timer(ti.recorder, info_copy.render_time_index);
                    }

                }));
}
